import json
from pathlib import Path


def test_colab_notebook_uses_public_repositories_and_cloudflare_tunnel():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "LTTS_demo.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "github.com/seruva19/LTTS.git" in source
    assert "github.com/seruva19/LTTS-modules.git" in source
    assert "start_quick_tunnel" in source
    assert "cloudflared-linux-amd64" in source
    assert "/api/ltts/modules" in source
    assert "localhost.run" not in source
    assert ".lhr.life" not in source
