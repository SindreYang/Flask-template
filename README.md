# Flask inference service template

A small Flask application factory with separated configuration and routes. The
included route demonstrates how to accept a mesh, call an inference backend,
and return generated files without relying on a developer-specific file path.

## Install and run

```bash
python -m pip install -r requirements.txt
python app.py
```

The service listens on port `5000`. Check that it started with:

```bash
curl http://127.0.0.1:5000/health
```

The response is `{"status":"ok"}`. Debug mode is off by default. Set
`CORS_ORIGINS` to the frontend origin when deploying, for example
`https://example.com`.

## Segmentation example

Send a `multipart/form-data` `POST` to `/split` with:

| Field | Description |
| --- | --- |
| `file` | Required PLY, OBJ, or STL mesh, up to 100 MB |
| `model` | One of the names configured in `conf/config.py` |
| `constraints` | Optional integer from 1 to 1,000,000; defaults to 200,000 |
| `refine_switch` | Optional `0` or `1`; defaults to `1` |

On success, the response contains a job ID and download URLs for the PLY, OBJ,
and MTL files that were generated. Each job has its own cache directory, so
parallel requests do not overwrite one another. Use
`GET /split/download/<job_id>/<artifact>` with `ply`, `obj`, or `mtl` as the
artifact name.

The model checkpoints and network implementation are not included in this
repository. Until they are installed, the application and health endpoint still
start normally, while inference requests return a clear `503` response.

## Tests

```bash
python -m unittest discover -s tests -v
# or, for older workflows:
python test.py
```

## Project layout

```text
app.py                 Flask application factory and health check
conf/config.py         App defaults and model checkpoint map
models/                Inference implementation and mesh utilities
routes/                Implemented HTTP endpoints
services/              Optional service integrations
tests/                 Isolated route and application tests
resources/cache/       Per-job upload and result files (created at runtime)
```
# 迁移说明

本模板已迁移并重构至 [`SindreTemplate`](https://github.com/SindreYang/SindreTemplate/tree/template/flask) 的 `template/flask` 分支。该仓库保留历史记录，不再作为新项目的推荐起点。
