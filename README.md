# ComfyUI OpenAI Adapter

Serve ComfyUI workflows through the **OpenAI Images API** — no bridge process, no extra port, no config file. The routes live inside ComfyUI's own `:8188`, so any OpenAI client (Snow App, Cherry Studio, your own script) can generate from a local graph by naming the workflow as the `model`.

```
POST http://127.0.0.1:8188/v1/images/generations
{
  "model": "my-workflow",
  "prompt": "a red apple on a wooden table",
  "size": "1024x1024",
  "params": { "7.steps": 30, "7.cfg": 3.5 }
}
```

---

把 ComfyUI 工作流暴露成 **OpenAI Images API**，让只认这套协议的上游（例如 Snow App 的生图渠道）直接调用。

**没有额外进程、没有额外端口、没有桥配置文件** —— 路由就在 ComfyUI 自己的 8188 里。

## 安装

### Git

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShiroEirin/ComfyUI-OpenAI-Adapter.git
```

重启 ComfyUI。只要 `GET http://127.0.0.1:8188/v1/models` 能返回工作流清单，就已经生效。

### Comfy Registry（发布后可用）

```bash
comfy node install comfyui-openai-adapter
```

## 注册一个工作流

放入 `workflows/` 一个文件就够：

```
workflows/
  my-workflow.json         API 格式工作流（就是 /prompt 能直接吃的 JSON）
  my-workflow.meta.json    可选，只补机器猜不到的信息
```

文件名（去掉 `.json`）就是 **模型 id**，客户端在 `model` 字段里用它。

仓库自带四个示例，覆盖三种模型族：`moss-image-layout`、`moss-image-layout-merge`、`qwen-t2i-base-bf16`、`anima-t2i`。它们只保存图谱，不含任何权重。

### 零配置：meta 可以不写

适配器会扫描图结构自动推导：

| 推导项 | 依据 |
|---|---|
| `prompt_node` / `prompt_field` | 主采样器 `positive` 连线向上找文本编码器的 `text` |
| `negative_node` | 主采样器 `negative` 连线，同法向上找 |
| `size_nodes` | 主采样器 `latent_image` 向上追溯，直到带 `width`/`height` 的节点（`Empty*LatentImage` 等） |
| `seed_nodes` | 主采样器自带 `seed` / `noise_seed` |
| `sampler_node` | 含 `steps` 的采样节点；多个时选 `latent_image` 直连 `Empty*LatentImage` 的那个（避开 hires 二段） |
| `layout_node` | class_type 含 `LayoutBoard` 的节点 |

实测：一份 qwen 图、一份 anima 图**都不放 meta**，推导出 `prompt=5 size=6 seed=8` 与 `prompt=11 size=28 seed=19`，与手写结果一致，并且直接出图成功。

`params_schema` 同样用 `/object_info` 自动生成（类型、默认值、min/max、枚举），所以节点新增一个输入，客户端下次就能看到它 —— 不需要改适配器。

### meta 字段（全部可选）

| 字段 | 说明 |
|---|---|
| `display_name` | 显示名（不填就用文件名） |
| `layout_node` / `prompt_node` / `prompt_field` | 显式指定，优先于推导 |
| `size_nodes` / `seed_nodes` | 显式指定，优先于推导 |
| `default_size` | 默认尺寸，如 `1024x1024` |
| `supported_ratios` / `supported_resolutions` | 给客户端做能力声明的字符串 |
| `timeout_seconds` | 单次执行超时，默认 600 |
| `params_schema` | 手写的参数说明（对象）；与自动生成合并，手写条目优先 |
| `aliases` | 自定义别名表 `{ "steps": "8.steps" }`，覆盖自动派生的同名项 |
| `negative_field` | 负向词写入的字段名，默认 `text` |
| `prompt_hint` | 给 Agent 的用法提示 |

### 导入已有工作流（推荐）

不用手写 meta。在 ComfyUI 里把图搭好，`Workflow → Export (API)` 导出 JSON，然后：

```bash
curl -X POST http://127.0.0.1:8188/v1/_import \
  -H "Content-Type: application/json" \
  -d '{"name":"my-workflow","workflow":{ ...导出的 JSON... }}'
```

适配器会：

1. 校验是不是 API 格式（顶层出现 `nodes`/`links` 直接拒绝，并提示去导出）
2. 扫描图结构自动推导 `prompt_node` / `size_nodes` / `seed_nodes` / `sampler_node` / `layout_node`
3. 用 `/object_info` 生成 `params_schema`
4. 原子写入 `workflows/<name>.json` 与 `workflows/<name>.meta.json`

响应里带 `routing`（推导落在哪些节点）与 `inferred`（哪些键是猜的），核对一眼就知道准不准。

`name` 只能含字母、数字、点、下划线、短横线，首字符不能是符号。同名默认拒绝，要覆盖传 `"overwrite": true`。想改推导结果，再传 `"meta": {...}`，其中字段会覆盖自动值。

## 接口

### `GET /v1/models`

每个工作流返回 `id` / `display_name` / `default_size` / `params_schema`，另带三个诊断字段：

- `routing` —— 这次真正写到了哪些节点（如 `prompt_node=5 size_nodes=6 seed_nodes=8`），用来证明推导没走错
- `inferred` —— 哪些键是扫描得出的、而不是 meta 写的
- `node_inputs` —— 全图每个节点的可写字段名，点号键的完整地图

### `POST /v1/images/generations`

```json
{
  "model": "my-workflow",
  "prompt": "画面描述",
  "size": "1024x1024",
  "seed": 42,
  "params": {
    "layout": { "width": 1024, "height": 1024, "items": [] },
    "strict": true,
    "8.steps": 30,
    "8.cfg": 5.0
  }
}
```

`params` 有两种等价写法，适配器都认：

1. **嵌套**：`{"params": {"layout": {...}}}`
2. **平铺**：`{"layout": {...}, "strict": true}`（Snow App 的工具层会把 `params` 拍平到请求体顶层）

参数定位规则：

- 普通键（如 `strict`、`character_boxes`）→ 写到 `layout_node` 的 `inputs`
- `"<节点id>.<字段>"`（如 `"8.cfg"`）→ 写到该节点的该字段；**只要节点类定义了这个输入就接受**，哪怕当前模板没写它
- `layout` → 覆盖布局文档，并同步 `size` 到布局文档、布局节点、`size_nodes` 三处；图里没有布局节点时进 `ignored_params`
- **别名**（`steps` / `cfg` / `sampler` / `scheduler` / `denoise` / `negative`）→ 由 `routing` 自动派生，等价于对应点号键；显式点号键优先
- 落到不存在、且节点类也没定义的字段时**不会静默丢弃**，会出现在响应的 `ignored_params` 里

### 负向提示词、种子与别名

| 字段 | 语义 |
|---|---|
| `negative_prompt` | 写到推断出的 `negative_node`；只有单个文本编码器的图会返回 `ignored_params` |
| `seed: -1` | 运行时随机（32 位无符号），实际值在响应 `seed` 字段回显，便于复现 |
| `seed` 越界 | 裁到 `0..4294967295`；负数（除 `-1`）裁到 `0` |
| `params: { "steps": 4 }` | 用别名，不必知道采样器是节点 8 |

`GET /v1/models` 的 `aliases` 字段列出当前模型的别名表，`params_schema` 里也镜像了同名条目（描述写 `alias → 8.steps`）。

### `GET /v1/_debug/last`

返回最近一次生成请求的字段摘要（不含图片数据），用于排查「参数到底有没有到」。

## 注意

- 工作流必须是 **API 格式**（ComfyUI 的 `/prompt` 接受的那种），不是 UI 编辑器保存的格式。
- 尺寸只传一次：适配器会同步布局节点宽高 / 布局文档内部宽高 / `size_nodes` 宽高，避免静默拉伸。
- `response_format`、`quality`、`stream` 等 OpenAI 标准字段不会被当成工作流参数转发。
- 图结构有歧义时（一张图多处采样、多个文本编码器），用 `/v1/models` 的 `routing` 核对；不对就在 meta 里显式写死。

## 开发与发布

- 只依赖标准库与 ComfyUI 自带的 `aiohttp`，没有第三方包（因此 `dependencies = []`）。
- 发布到 [Comfy Registry](https://registry.comfy.org) 前，先在注册页创建 publisher，把 `pyproject.toml` 里 `[tool.comfy] PublisherId` 填成个人主页 `@` 后面那段 id，然后 `comfy node publish`。
- 也可以添加 `.github/workflows/publish_action.yml`，在 `pyproject.toml` 版本变更时自动发布。

## License

MIT
