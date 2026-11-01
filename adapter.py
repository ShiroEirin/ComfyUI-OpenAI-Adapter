"""OpenAI-compatible image endpoints served by ComfyUI itself.

Why this exists: Snow App's image generation tool only speaks the OpenAI Images API
(``POST {baseUrl}/images/generations``), so a separate bridge process used to be the
only way to reach a local workflow. ComfyUI can host those routes itself, which drops
the bridge from the chain and lets the caller pass workflow-specific parameters that
the bridge could never express.

    GET  /v1/models                list workflows as models, with their parameter schema
    POST /v1/images/generations    run a workflow, return {data:[{b64_json}]}

A workflow becomes a model by dropping two files into ./workflows:

    <name>.json        API-format workflow (the same JSON ComfyUI's /prompt accepts)
    <name>.meta.json   layout_node / prompt_field / size_nodes / seed_nodes / params_schema

Only the standard library plus aiohttp (which ComfyUI already ships) is used.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
import uuid
from pathlib import Path

WORKFLOW_DIR = Path(__file__).with_name("workflows")

# A model id becomes a file name, so it must not be able to escape WORKFLOW_DIR, hide as
# a dotfile, or collide with the ``<name>.meta.json`` sidecar convention.
WORKFLOW_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

# Size must only travel through `size`: a flattened `width`/`height` would bypass the
# three-way synchronization below and leave the graph internally inconsistent.
SIZE_KEYS = ("width", "height")

# Spaces whose coordinates are ratios, not pixels — rescaling a canvas must not touch them.
NORM_SPACES = {"norm_1000", "relative_1000", "normalized_1000", "0_1000"}

# Layout-board inputs that a template may legitimately leave out (ComfyUI then falls
# back to the node defaults). They must still be settable through params, otherwise
# `strict=True` - the one flag that turns a degraded layout into a real error - would
# be dropped in silence on every template that does not spell it out.
LAYOUT_OPTIONAL_INPUTS = {"character_boxes", "on_overflow", "strict"}

# OpenAI's own request fields: understood here, never forwarded blindly.
# Snow App flattens a tool's ``params`` object into the body root, so every field the
# Images API defines must be listed here or a normal ``quality``/``stream`` flag would
# be mistaken for a workflow parameter.
RESERVED_BODY = {
    "model", "prompt", "n", "response_format", "images", "seed", "size", "params",
    "quality", "output_format", "output_compression", "background", "moderation",
    "stream", "partial_images", "aspect_ratio", "resolution", "user",
}

# The graph is scanned instead of described, so a brand-new workflow lands in
# /v1/models with correct parameters and no hand-written meta sidecar. These are the
# only structural hints used: a node that samples, a node that holds text, a node that
# holds a latent canvas.
SAMPLER_HINTS = ("KSampler", "SamplerCustom")
TEXT_HINTS = ("CLIPTextEncode", "TextEncode")

# Inputs that are already mapped to a first-class API field (prompt / size / seed) or
# that carry a wire rather than a value — never offered as a dotted parameter.
SCHEMA_SKIP_FIELDS = {
    "seed", "noise_seed", "width", "height", "text", "clip", "model", "vae",
    "positive", "negative", "latent_image", "samples", "image", "pixels",
    "mask", "guider", "sampler", "sigmas", "noise",
}

# ComfyUI ships ~70 sampler names. Enumerating them all would drown the description a
# caller actually reads, so past this size the schema says "ask /object_info" instead.
ENUM_DISPLAY_LIMIT = 16

# /object_info for one class is fixed for the life of the process: cache it so a 30-node
# graph costs one round trip per distinct class and repeated /v1/models calls stay cheap.
_OBJECT_INFO_CACHE: dict[str, dict] = {}

DEFAULT_TIMEOUT = 600.0

# Diagnostic slot: the last /v1/images/generations body, so a caller can prove what
# actually arrived (a dropped `params` looks exactly like a template-driven image).
LAST_REQUEST: dict = {}


def _load_meta(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_workflows() -> dict[str, dict]:
    """Map model id -> {workflow, meta}. A broken file is skipped, never fatal."""
    found: dict[str, dict] = {}
    if not WORKFLOW_DIR.is_dir():
        return found
    for path in sorted(WORKFLOW_DIR.glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        try:
            workflow = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(workflow, dict) or not workflow:
            continue
        meta = _load_meta(path.with_suffix(".meta.json"))
        meta.setdefault("display_name", path.stem)
        found[path.stem] = {"workflow": workflow, "meta": meta}
    return found


def _link(value) -> str | None:
    """A ComfyUI wire is ``[node_id, output_slot]``; a literal value is not."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        return value[0]
    return None


def _input_spec(object_info: dict, class_type: str, field: str):
    """Look one input up in /object_info, required first then optional."""
    node_def = object_info.get(class_type) or {}
    for group in ("required", "optional"):
        spec = (node_def.get("input") or {}).get(group) or {}
        if field in spec:
            return spec[field]
    return None


async def _get_object_info(session, base: str, class_type: str) -> dict:
    """Fetch and cache one node definition.

    A miss is cached as ``{}`` too, so an unregistered class costs one round trip for the
    life of the process instead of one per request.
    """
    if class_type in _OBJECT_INFO_CACHE:
        return _OBJECT_INFO_CACHE[class_type]
    try:
        from urllib.parse import quote
        data = await _upstream(session, base, f"/object_info/{quote(class_type, safe='')}",
                              timeout=20.0)
        node_def = (data or {}).get(class_type) or {}
    except Exception:
        node_def = {}
    _OBJECT_INFO_CACHE[class_type] = node_def if isinstance(node_def, dict) else {}
    return _OBJECT_INFO_CACHE[class_type]


async def _load_object_info(workflows: dict) -> dict:
    """Warm the cache for every class a workflow set uses.

    Never raises: inference degrades to pure structural hints when /object_info is
    unreachable, and the templates' own fields still work as before.
    """
    classes = {
        str(node.get("class_type"))
        for entry in workflows.values()
        for node in (entry.get("workflow") or {}).values()
        if isinstance(node, dict) and node.get("class_type")
    }
    missing = sorted(cls for cls in classes if cls not in _OBJECT_INFO_CACHE)
    if not missing:
        return _OBJECT_INFO_CACHE
    try:
        import aiohttp
        from server import PromptServer
        base = f"http://127.0.0.1:{PromptServer.instance.port}"
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*[_get_object_info(session, base, cls) for cls in missing])
    except Exception:
        pass
    return _OBJECT_INFO_CACHE


def _upstream_text_node(workflow: dict, start_id: str | None) -> str | None:
    """Walk upstream from a conditioning wire to the node that actually holds text.

    ``positive`` frequently points at a combine/conditioning node rather than the
    encoder, so a single field lookup is not enough.
    """
    seen: set[str] = set()
    queue = [start_id] if start_id else []
    while queue and len(seen) < 80:
        node_id = queue.pop(0)
        if not node_id or node_id in seen or node_id not in workflow:
            continue
        seen.add(node_id)
        node = workflow[node_id]
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        if any(hint in str(node.get("class_type") or "") for hint in TEXT_HINTS) \
                and isinstance(inputs.get("text"), str):
            return node_id
        for value in inputs.values():
            upstream = _link(value)
            if upstream and upstream not in seen:
                queue.append(upstream)
    return None


def _infer_meta(workflow: dict) -> dict:
    """Derive the sidecar meta from the graph itself.

    This is what makes a new workflow zero-maintenance: dropping the API JSON into
    ./workflows is enough, and an explicit .meta.json only carries what no heuristic can
    know (display text, supported ratios, timeout, tuned prompt hints).
    """
    inferred: dict = {}
    samplers = [
        node_id for node_id, node in workflow.items()
        if isinstance(node, dict) and (
            "steps" in (node.get("inputs") or {})
            or any(hint in str(node.get("class_type") or "") for hint in SAMPLER_HINTS)
        )
    ]
    primary = None
    for node_id in samplers:
        # A hires/refine pass samples an upscaled latent; the primary sampler is the one
        # whose latent_image still comes straight from an Empty*Latent node.
        latent = _link((workflow[node_id].get("inputs") or {}).get("latent_image"))
        cls = str((workflow.get(latent) or {}).get("class_type") or "") if latent else ""
        if "Empty" in cls and "Latent" in cls:
            primary = node_id
            break
    primary = primary or (samplers[0] if samplers else None)
    if primary:
        inferred["sampler_node"] = primary
        inputs = workflow[primary].get("inputs") or {}
        if isinstance(inputs.get("seed"), int) or isinstance(inputs.get("noise_seed"), int):
            inferred["seed_nodes"] = [primary]
        text_node = _upstream_text_node(workflow, _link(inputs.get("positive")))
        if text_node:
            inferred["prompt_node"] = text_node
            inferred["prompt_field"] = "text"
        negative = _upstream_text_node(workflow, _link(inputs.get("negative")))
        if negative and negative != text_node:
            inferred["negative_node"] = negative
        # The canvas comes from the latent feeding the primary sampler; follow the chain
        # through VAEEncode so an img2img graph is sized at the source too.
        size_nodes: list[str] = []
        cursor = _link(inputs.get("latent_image"))
        for _ in range(8):
            if not cursor or cursor not in workflow:
                break
            wired = workflow[cursor].get("inputs") or {}
            if "width" in wired and "height" in wired:
                size_nodes.append(cursor)
                break
            cursor = _link(wired.get("samples")) or _link(wired.get("image"))
        if not size_nodes:
            for node_id, node in workflow.items():
                cls = str(node.get("class_type") or "") if isinstance(node, dict) else ""
                if "Empty" in cls and "Latent" in cls and "width" in (node.get("inputs") or {}):
                    size_nodes.append(node_id)
                    break
        if size_nodes:
            inferred["size_nodes"] = size_nodes
    if "prompt_node" not in inferred:
        # A layout board carries the prompt itself; nothing else in such a graph does.
        for node_id, node in workflow.items():
            if isinstance(node, dict) and "LayoutBoard" in str(node.get("class_type") or ""):
                if "prompt" in (node.get("inputs") or {}):
                    inferred["prompt_node"] = node_id
                    inferred["prompt_field"] = "prompt"
                break
    return inferred


def _merge_meta(explicit: dict, inferred: dict) -> dict:
    """Explicit sidecar fields win; the graph fills in everything left unset."""
    merged = dict(inferred)
    for key, value in explicit.items():
        if value is not None:
            merged[key] = value
    merged.setdefault("display_name", "")
    return merged


def _schema_entry(class_type: str, field: str, spec) -> dict:
    """One /object_info input spec -> a compact entry a caller can act on."""
    entry: dict = {"type": "string", "description": f"{class_type}.{field}"}
    if not isinstance(spec, list) or not spec:
        return entry
    kind = spec[0]
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(kind, list):
        if 0 < len(kind) <= ENUM_DISPLAY_LIMIT:
            entry["enum"] = [str(item) for item in kind]
        else:
            entry["description"] = (
                f"{class_type}.{field}（{len(kind)} 个可选值，完整列表见 /object_info）")
        return entry
    entry["type"] = {
        "INT": "integer", "FLOAT": "number", "STRING": "string",
        "BOOLEAN": "boolean", "COMBO": "string",
    }.get(str(kind).upper(), "string")
    if "default" in options:
        entry["default"] = options["default"]
    for bound in ("min", "max"):
        if bound in options:
            entry[bound] = options[bound]
    return entry


def _build_param_schema(workflow: dict, meta: dict, object_info: dict) -> dict:
    """Generate params_schema from the graph so it stays right as nodes evolve.

    Hand-written entries are kept verbatim (they carry better prose); the generator only
    fills keys nobody described yet. Dotted keys are accepted at runtime regardless, so a
    field this list happens to miss is still settable.
    """
    explicit = meta.get("params_schema")
    schema: dict = dict(explicit) if isinstance(explicit, dict) else {}
    targets: list[str] = []
    for key in ("sampler_node", "layout_node"):
        if meta.get(key):
            targets.append(str(meta[key]))
    for key in ("size_nodes", "seed_nodes"):
        targets.extend(str(node_id) for node_id in (meta.get(key) or []))
    for node_id in dict.fromkeys(targets):
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        node_def = object_info.get(class_type) or {}
        declared: dict = {}
        for group in ("required", "optional"):
            declared.update((node_def.get("input") or {}).get(group) or {})
        for field, value in (node.get("inputs") or {}).items():
            if field in SCHEMA_SKIP_FIELDS or _link(value):
                # A wired input is a connection, not a value: writing it breaks the graph.
                continue
            if object_info and field not in declared:
                continue
            schema.setdefault(f"{node_id}.{field}",
                              _schema_entry(class_type, field, declared.get(field)))
    return schema


def _node_inputs(workflow: dict, object_info: dict) -> dict:
    """Every writable scalar in the graph, by node — the map behind dotted keys.

    Field names only (no specs): it answers "which knobs live on node 12?" compactly,
    while /object_info still holds the types and ranges.
    """
    summary: dict = {}
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        fields = []
        for field, value in (node.get("inputs") or {}).items():
            if field in SCHEMA_SKIP_FIELDS or _link(value):
                continue
            if object_info and _input_spec(object_info, class_type, field) is None:
                continue
            fields.append(field)
        if fields:
            summary[str(node_id)] = {"class": class_type, "fields": sorted(fields)}
    return summary


def _resolve_layout_node(workflow: dict, meta: dict) -> str | None:
    """Pick the node whose inputs carry prompt/layout.

    ``layout_node`` wins; otherwise the first node that looks like a layout board.
    """
    node_id = meta.get("layout_node")
    if node_id and node_id in workflow:
        return str(node_id)
    for key, node in workflow.items():
        # Heuristic only: prefer an explicit `layout_node` in the sidecar meta above.
        # Any node whose class name carries "LayoutBoard" qualifies, so this adapter
        # stays independent of the specific layout plugin installed.
        if isinstance(node, dict) and "LayoutBoard" in str(node.get("class_type") or ""):
            return str(key)
    return None


def _coerce_layout(value) -> dict | None:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return dict(parsed) if isinstance(parsed, dict) else None
    return None


def _parse_size(raw) -> tuple[int, int] | None:
    if not isinstance(raw, str):
        return None
    try:
        width, height = (int(part.strip()) for part in raw.lower().split("x", 1))
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _rescale_boxes(layout: dict, old: tuple[int, int], new: tuple[int, int]) -> bool:
    """Scale pixel boxes with the canvas they were authored on.

    Changing only ``width``/``height`` leaves every ``items[].box`` on the old grid, so
    the graph still runs and still returns a picture - the composition is just silently
    shifted and shrunk. Ratio spaces are canvas-independent and skipped.
    """
    space = str(layout.get("bbox_coord_space") or "").lower()
    if space in NORM_SPACES or not all(old) or old == new:
        return False
    y_first = str(layout.get("bbox_format") or "").lower() == "yxyx"
    sx, sy = new[0] / old[0], new[1] / old[1]
    moved = False
    for item in layout.get("items") or []:
        if not isinstance(item, dict):
            continue
        key = "box" if "box" in item else ("bbox" if "bbox" in item else None)
        if key is None:
            continue
        box = item.get(key)
        if not isinstance(box, list) or len(box) != 4:
            continue
        scaled = []
        for index, value in enumerate(box):
            if not isinstance(value, (int, float)):
                scaled = []
                break
            # yxyx: indices 0/2 are y, 1/3 are x.
            scaled.append(round(value * (sy if y_first == (index % 2 == 0) else sx)))
        if len(scaled) == 4:
            item[key] = scaled
            moved = True
    return moved


def _apply_size(workflow: dict, meta: dict, layout_node: str | None,
                width: int, height: int) -> list[str]:
    """Keep every size in the graph agreeing: board, latent, and the layout document.

    A mismatch here is the classic silent failure - the picture comes out, the boxes
    are simply stretched - so all three are written together, and the boxes are
    rescaled from the document's own canvas to the requested one.
    """
    touched: list[str] = []
    if layout_node and layout_node in workflow:
        inputs = workflow[layout_node].setdefault("inputs", {})
        inputs["width"] = width
        inputs["height"] = height
        touched.append(f"{layout_node}.width/height")

        layout = _coerce_layout(inputs.get("layout"))
        if layout is not None:
            old = (int(layout.get("width") or 0), int(layout.get("height") or 0))
            moved = _rescale_boxes(layout, old, (width, height))
            if old != (width, height) or moved:
                layout["width"] = width
                layout["height"] = height
                inputs["layout"] = json.dumps(layout, ensure_ascii=False)
                touched.append(
                    f"{layout_node}.layout(width,height{'+boxes' if moved else ''})")

    for node_id in meta.get("size_nodes") or []:
        node = workflow.get(str(node_id))
        if isinstance(node, dict):
            node.setdefault("inputs", {}).update({"width": width, "height": height})
            touched.append(f"{node_id}.width/height")
    return touched


def _apply_extra_params(workflow: dict, layout_node: str | None,
                        params: dict, object_info: dict | None = None
                        ) -> tuple[list[str], list[str]]:
    """Merge caller-supplied workflow parameters.

    Two forms are accepted so a caller never has to know the whole graph:

    * ``"strict": true``          -> an input on the layout node
    * ``"7.steps": 30``           -> ``<node id>.<input>`` anywhere in the graph

    A field the node *class* declares is written even when this template leaves it out —
    ComfyUI then uses the caller's value over the node default, which is what makes an
    unlisted-but-real knob reachable. Only a field the class does not define at all is
    reported back as ignored.
    """
    applied: list[str] = []
    ignored: list[str] = []
    info = object_info or {}
    for key, value in params.items():
        if not isinstance(key, str) or not key:
            continue
        if "." in key:
            node_id, _, field = key.partition(".")
            node = workflow.get(node_id)
            inputs = node.get("inputs") if isinstance(node, dict) else None
            if not isinstance(inputs, dict):
                ignored.append(key)
                continue
            if field not in inputs:
                class_type = str(node.get("class_type") or "")
                if _input_spec(info, class_type, field) is None:
                    # No object_info to vouch for the field: keep the strict old behaviour
                    # rather than writing a typo into the graph.
                    ignored.append(key)
                    continue
            inputs[field] = value
            applied.append(key)
            continue
        if not layout_node or layout_node not in workflow:
            ignored.append(key)
            continue
        inputs = workflow[layout_node].setdefault("inputs", {})
        if key not in inputs and key not in LAYOUT_OPTIONAL_INPUTS:
            class_type = str(workflow[layout_node].get("class_type") or "")
            if _input_spec(info, class_type, key) is None:
                ignored.append(key)
                continue
        inputs[key] = value
        applied.append(key)
    return applied, ignored


def _describe(model_id: str, entry: dict, object_info: dict | None = None) -> dict:
    """Describe one workflow as a model, with its parameters derived from the graph.

    A ``.meta.json`` is optional now: whatever it leaves unset is inferred, so a freshly
    dropped workflow appears with usable controls and no hand-written schema. The
    ``inferred`` field names the keys that came from scanning rather than from the file.
    """
    meta = entry["meta"]
    workflow = entry.get("workflow") or {}
    info = object_info or {}
    inferred = _infer_meta(workflow)
    merged = _merge_meta(meta, inferred)
    layout_node = _resolve_layout_node(workflow, merged)
    if layout_node:
        merged.setdefault("layout_node", layout_node)
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "comfyui",
        "display_name": merged.get("display_name") or model_id,
        "params_schema": _build_param_schema(workflow, merged, info),
        "node_inputs": _node_inputs(workflow, info),
        "inferred": sorted(key for key in inferred if key not in meta),
        # Which nodes the routing actually landed on: proving an inference from the model
        # list is what keeps "it silently used the wrong encoder" from being invisible.
        "routing": {key: merged.get(key) for key in (
            "prompt_node", "prompt_field", "size_nodes", "seed_nodes",
            "sampler_node", "layout_node", "negative_node") if merged.get(key)},
        "default_size": merged.get("default_size") or "",
        "default_quality": merged.get("default_quality") or "",
        "supported_ratios": merged.get("supported_ratios") or "",
        "supported_resolutions": merged.get("supported_resolutions") or "",
        "notes": merged.get("prompt_hint") or "",
    }


def _write_json_atomic(path: Path, payload) -> None:
    """Write through a sibling temp file so a crash never leaves a half-written workflow."""
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _normalize_api_workflow(raw) -> dict:
    """Validate the shape ``/prompt`` accepts, and drop export-only metadata.

    ComfyUI's editor format (``nodes`` / ``links``) is rejected with a pointed message:
    the UI→API conversion lives in the frontend, so the caller has to run
    Workflow → Export (API) before importing.
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError("workflow 必须是非空对象")
    if "nodes" in raw or "links" in raw:
        raise ValueError(
            "这是 UI 编辑器格式（顶层含 nodes/links）。请在 ComfyUI 里用 "
            "Workflow → Export (API) 导出，再导入导出的 JSON。")
    workflow: dict = {}
    for node_id, node in raw.items():
        if node_id == "_meta":
            continue
        if not isinstance(node, dict) or not node.get("class_type"):
            raise ValueError(f"节点 {node_id} 不是 API 格式：缺少 class_type")
        workflow[str(node_id)] = node
    if not workflow:
        raise ValueError("workflow 里没有任何节点")
    return workflow


def _compose_import_meta(name: str, workflow: dict, inferred: dict,
                         object_info: dict) -> dict:
    """Build the sidecar for an imported graph out of what the graph itself says."""
    meta: dict = {"display_name": name, "timeout_seconds": 560}
    meta.update({key: value for key, value in inferred.items() if value})
    layout_node = _resolve_layout_node(workflow, meta)
    if layout_node:
        meta["layout_node"] = layout_node
    # A literal canvas on the size node is the only trustworthy default: anything else
    # would advertise a size this graph never actually uses.
    size_nodes = meta.get("size_nodes") or []
    if size_nodes:
        inputs = (workflow.get(str(size_nodes[0])) or {}).get("inputs") or {}
        width, height = inputs.get("width"), inputs.get("height")
        if isinstance(width, int) and isinstance(height, int):
            meta["default_size"] = f"{width}x{height}"
    meta["params_schema"] = _build_param_schema(workflow, meta, object_info)
    return meta


def _error(status: int, message: str, code: str = "invalid_request_error") -> tuple[int, dict]:
    return status, {"error": {"message": message, "type": code, "code": code}}


async def _upstream(session, base: str, path: str, payload=None, timeout=60.0):
    import aiohttp

    url = base.rstrip("/") + path
    async with session.request(
        "POST" if payload is not None else "GET",
        url,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as response:
        body = await response.read()
        if response.status >= 400:
            raise RuntimeError(f"{path} -> HTTP {response.status}: {body[:400]!r}")
        return json.loads(body.decode("utf-8")) if body else {}


async def _run_workflow(workflow: dict, client_id: str, timeout: float) -> dict:
    """Queue the graph on this same ComfyUI instance and wait for its outputs."""
    import aiohttp
    from server import PromptServer

    instance = PromptServer.instance
    base = f"http://127.0.0.1:{instance.port}"
    deadline = time.time() + timeout

    async with aiohttp.ClientSession() as session:
        queued = await _upstream(session, base, "/prompt",
                                 {"prompt": workflow, "client_id": client_id})
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI 没有返回 prompt_id：{queued}")

        history = None
        while time.time() < deadline:
            data = await _upstream(session, base, f"/history/{prompt_id}")
            history = data.get(prompt_id)
            if history:
                break
            await asyncio.sleep(1.0)

        if not history:
            # An abandoned prompt keeps its slot in the queue, so every later request
            # waits behind a job nobody is listening for. Drop it before failing.
            try:
                await _upstream(session, base, "/queue", {"delete": [prompt_id]})
            except Exception:
                pass
            raise RuntimeError(
                f"{int(timeout)}s 内没有完成，已尝试从队列移除 prompt_id={prompt_id}")

        status = history.get("status") or {}
        if status.get("status_str") == "error":
            raise RuntimeError(_node_error(status))

        outputs = history.get("outputs") or {}
        images = []
        for node_output in outputs.values():
            for image in node_output.get("images") or []:
                if image.get("type") not in (None, "output", "temp"):
                    continue
                query = (
                    f"/view?filename={image.get('filename', '')}"
                    f"&subfolder={image.get('subfolder', '')}"
                    f"&type={image.get('type', 'output')}"
                )
                async with session.get(base + query) as response:
                    if response.status >= 400:
                        continue
                    images.append(await response.read())
        if not images:
            raise RuntimeError("工作流跑完了，但没有产出可读取的图片")
        return {"images": images, "prompt_id": prompt_id}


def _node_error(status: dict) -> str:
    for entry in status.get("messages") or []:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        kind, payload = entry[0], entry[1]
        if kind != "execution_error" or not isinstance(payload, dict):
            continue
        node_id = payload.get("node_id")
        node_type = payload.get("node_type")
        message = payload.get("exception_message") or "执行失败"
        return f"节点 {node_id} ({node_type}) 报错：{message.strip()}"
    return "工作流执行失败（ComfyUI 没有给出详细信息）"


def register_routes() -> bool:
    try:
        from aiohttp import web
        from server import PromptServer
        routes = PromptServer.instance.routes
    except Exception:  # outside a running ComfyUI the node must still import
        return False

    @routes.get("/v1/models")
    async def list_models(request):
        workflows = load_workflows()
        object_info = await _load_object_info(workflows)
        return web.json_response({
            "object": "list",
            "data": [_describe(model_id, entry, object_info)
                     for model_id, entry in workflows.items()],
        })

    @routes.get("/v1/_debug/last")
    async def last_generation(request):
        return web.json_response(LAST_REQUEST)

    @routes.post("/v1/_import")
    async def import_workflow(request):
        """Adopt one exported graph as a model, inferring its sidecar on the way in."""
        try:
            body = await request.json()
        except Exception:
            status, payload = _error(400, "请求不是 JSON")
            return web.json_response(payload, status=status)

        name = str(body.get("name") or "").strip()
        if not WORKFLOW_NAME_RE.fullmatch(name):
            status, payload = _error(
                400, "name 只能用字母、数字、点、下划线、短横线，首字符不能是符号，最多 64 个字符")
            return web.json_response(payload, status=status)

        try:
            workflow = _normalize_api_workflow(body.get("workflow"))
        except ValueError as error:
            status, payload = _error(400, str(error))
            return web.json_response(payload, status=status)

        path = WORKFLOW_DIR / f"{name}.json"
        if path.exists() and not body.get("overwrite"):
            status, payload = _error(409, f"{name} 已存在；要覆盖请传 overwrite=true")
            return web.json_response(payload, status=status)

        object_info = await _load_object_info({name: {"workflow": workflow, "meta": {}}})
        inferred = _infer_meta(workflow)
        meta = _compose_import_meta(name, workflow, inferred, object_info)
        overrides = body.get("meta")
        if isinstance(overrides, dict):
            # Caller-supplied fields win, including a hand-written schema.
            meta.update({key: value for key, value in overrides.items()
                         if value is not None})

        WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(path, workflow)
        _write_json_atomic(WORKFLOW_DIR / f"{name}.meta.json", meta)

        return web.json_response({
            "ok": True,
            "id": name,
            "path": str(path),
            "routing": {key: meta.get(key) for key in (
                "prompt_node", "prompt_field", "size_nodes", "seed_nodes",
                "sampler_node", "layout_node", "negative_node") if meta.get(key)},
            "inferred": sorted(key for key in inferred
                               if key not in (overrides or {})),
            "default_size": meta.get("default_size") or "",
            "params_schema": meta.get("params_schema") or {},
            "node_inputs": _node_inputs(workflow, object_info),
        })

    @routes.post("/v1/images/generations")
    async def generate(request):
        try:
            body = await request.json()
        except Exception:
            status, payload = _error(400, "请求不是 JSON")
            return web.json_response(payload, status=status)

        # Keep the last body so a caller can prove which fields actually arrived.
        LAST_REQUEST.clear()
        LAST_REQUEST.update({
            "model": body.get("model"),
            "keys": sorted(body.keys()),
            "size": body.get("size"),
            "seed": body.get("seed"),
            "has_params": isinstance(body.get("params"), dict),
            "param_keys": sorted((body.get("params") or {}).keys())
            if isinstance(body.get("params"), dict) else [],
            "param_layout_items": len(((body.get("params") or {}).get("layout") or {}).get("items") or [])
            if isinstance((body.get("params") or {}).get("layout"), dict) else None,
        })

        model_id = str(body.get("model") or "").strip()
        if not model_id:
            status, payload = _error(400, "缺少 model；可用模型见 GET /v1/models")
            return web.json_response(payload, status=status)

        # A workflow run produces exactly one image. Returning one picture for `n=4`
        # with a success status is the kind of silent contract break that costs an hour
        # to notice, so an unsupported count is rejected instead of ignored.
        requested_n = body.get("n")
        if requested_n is not None:
            if isinstance(requested_n, bool) or not isinstance(requested_n, int) or requested_n < 1:
                status, payload = _error(400, f"n 必须是正整数，收到 {requested_n!r}")
                return web.json_response(payload, status=status)
            if requested_n > 1:
                status, payload = _error(
                    400, f"该工作流一次只出一张图；n={requested_n} 不支持，请分开多次调用")
                return web.json_response(payload, status=status)

        workflows = load_workflows()
        entry = workflows.get(model_id)
        if not entry:
            status, payload = _error(
                404, f"没有名为 {model_id} 的工作流；可用：{', '.join(sorted(workflows)) or '（workflows 目录为空）'}")
            return web.json_response(payload, status=status)

        object_info = await _load_object_info({model_id: entry})

        workflow = json.loads(json.dumps(entry["workflow"]))  # deep copy: templates stay clean
        # No sidecar is required any more: infer whatever the .meta.json leaves unset, so an
        # unfamiliar graph still routes prompt / size / seed to the right nodes.
        meta = _merge_meta(entry["meta"], _infer_meta(workflow))
        layout_node = _resolve_layout_node(workflow, meta)
        if layout_node:
            meta.setdefault("layout_node", layout_node)
        touched: list[str] = []
        ignored: list[str] = []

        prompt = body.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            # Three shapes: an explicit `prompt_node` (plain txt2img graphs), a layout
            # board (its own `prompt` input), or nothing to write to.
            if meta.get("prompt_node"):
                prompt_node = str(meta["prompt_node"])
                field = meta.get("prompt_field") or "text"
            else:
                prompt_node = str(layout_node or "")
                field = meta.get("prompt_field") or "prompt"
            if prompt_node and prompt_node in workflow:
                workflow[prompt_node].setdefault("inputs", {})[field] = prompt
                touched.append(f"{prompt_node}.{field}")
            else:
                ignored.append("prompt")

        # Snow App flattens the tool's `params` object into the request body root
        # (layout / strict / "7.steps" arrive as siblings of prompt), while a direct
        # API caller may nest them under `params`. Accept both shapes.
        params = body.get("params")
        params = dict(params) if isinstance(params, dict) else {}
        for key, value in body.items():
            if key in RESERVED_BODY:
                continue
            params.setdefault(key, value)

        # Size only travels through `size`: a flattened width/height would change the
        # board alone and leave the latent and the layout document disagreeing.
        for key in SIZE_KEYS:
            if key in params:
                params.pop(key)
                ignored.append(key)

        # A caller may hand the whole layout document over through params.layout;
        # it wins over the template so one workflow can serve many compositions.
        layout_override = params.pop("layout", None)
        if layout_override is not None and layout_node and layout_node in workflow:
            layout = _coerce_layout(layout_override)
            if layout is None:
                status, payload = _error(400, "params.layout 既不是对象也不是可解析的 JSON")
                return web.json_response(payload, status=status)
            workflow[layout_node].setdefault("inputs", {})["layout"] = json.dumps(
                layout, ensure_ascii=False)
            touched.append(f"{layout_node}.layout")

        size = _parse_size(body.get("size")) or _parse_size(meta.get("default_size"))
        if size:
            touched.extend(_apply_size(workflow, meta, layout_node, size[0], size[1]))

        seed = body.get("seed")
        if isinstance(seed, int):
            for node_id in meta.get("seed_nodes") or []:
                node = workflow.get(str(node_id))
                if isinstance(node, dict):
                    node.setdefault("inputs", {})["seed"] = seed
                    touched.append(f"{node_id}.seed")

        applied, rejected = _apply_extra_params(workflow, layout_node, params, object_info)
        touched.extend(applied)
        ignored.extend(rejected)

        timeout = float(meta.get("timeout_seconds") or DEFAULT_TIMEOUT)
        try:
            result = await _run_workflow(workflow, f"openai-compat-{uuid.uuid4()}", timeout)
        except Exception as error:
            status, payload = _error(500, str(error), "generation_failed")
            return web.json_response(payload, status=status)

        return web.json_response({
            "created": int(time.time()),
            "model": model_id,
            "prompt_id": result["prompt_id"],
            "applied": touched,
            "ignored_params": ignored,
            "data": [{"b64_json": base64.b64encode(image).decode("ascii")}
                     for image in result["images"]],
        })

    return True
