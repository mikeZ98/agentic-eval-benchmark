"""Golden Dataset loaders: JSON, JSONL, Parquet, CSV, aliases, malformed input, reference set."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agentic_eval_benchmark.schemas import GoldenDataset
from agentic_eval_benchmark.tools import dataset_loader
from agentic_eval_benchmark.tools.dataset_loader import (
    REFERENCE_DATASET_NAME,
    DatasetLoadError,
    build_dataset,
    detect_format,
    load_golden_dataset,
    load_reference_dataset,
)

RAGAS_ROWS = [
    {
        "id": 1,
        "user_input": "What is X?",
        "reference_contexts": ["X is Y."],
        "reference": "X is Y.",
    },
    {
        "id": 2,
        "user_input": "What is Z?",
        "reference_contexts": ["Z is W."],
        "reference": "Z is W.",
        "metadata": {"k": 1},
    },
]


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_reference_dataset_is_valid_and_diverse() -> None:
    reference = load_reference_dataset()
    scenarios = {s.metadata["scenario"] for s in reference.samples}
    assert reference.name == REFERENCE_DATASET_NAME
    assert len(reference.samples) == 8
    assert len(scenarios) == 8
    assert {"multi_hop_reasoning", "negation_exception", "numerical_limit"} <= scenarios


def test_json_document_round_trip(tmp_path: Path, reference_dataset: GoldenDataset) -> None:
    path = write(tmp_path / "doc.json", reference_dataset.to_json())
    assert load_golden_dataset(path) == reference_dataset
    assert load_golden_dataset(path, name=reference_dataset.name) == reference_dataset


def test_bare_json_rows_with_ragas_aliases(tmp_path: Path) -> None:
    loaded = load_golden_dataset(write(tmp_path / "ragas style.json", json.dumps(RAGAS_ROWS)))
    assert loaded.name == "ragas-style"
    assert loaded.dataset_version == "0.0.0"
    assert loaded.sample_ids == ("1", "2")
    assert loaded.samples[1].metadata == {"k": 1}


def test_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    text = "\n".join(json.dumps(r) for r in RAGAS_ROWS) + "\n\n"
    loaded = load_golden_dataset(
        write(tmp_path / "rows.jsonl", text), name="x", dataset_version="2.0.0", description="d"
    )
    assert (loaded.name, loaded.dataset_version, loaded.description) == ("x", "2.0.0", "d")


CSV = (
    "question,contexts,ground_truth,source,metadata\n"
    '"What, exactly, is X?","[""X is Y."", ""Y is Z.""]",X is Y.,wiki,"{""lang"": ""en""}"\n'
    'What is Q?,"Q is R.\nIt spans lines.",Q is R.,kb,\n'
)


def test_csv_quoting_json_cells_and_extra_columns(tmp_path: Path) -> None:
    path = write(tmp_path / "rows.csv", CSV)
    loaded = load_golden_dataset(path, extra="metadata")
    first, second = loaded.samples
    assert first.contexts == ("X is Y.", "Y is Z.")
    assert first.metadata == {"lang": "en", "source": "wiki"}
    assert second.contexts == ("Q is R.\nIt spans lines.",)
    assert loaded.sample_ids == ("sample-0001", "sample-0002")
    assert load_golden_dataset(path, extra="ignore").samples[0].metadata == {"lang": "en"}
    assert load_golden_dataset(path, extra="ignore", id_prefix="q").sample_ids[0] == "q-0001"
    with pytest.raises(DatasetLoadError, match="unmapped fields"):
        load_golden_dataset(path)


def test_parquet_lists_and_structs(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": "p1",
            "question": "What is P?",
            "contexts": ["P is Q."],
            "ground_truth": "P is Q.",
            "metadata": {"lang": "en"},
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "rows.parquet")
    loaded = load_golden_dataset(tmp_path / "rows.parquet", name="pq", dataset_version="1.0.0")
    assert loaded.samples[0].contexts == ("P is Q.",)
    assert loaded.samples[0].metadata == {"lang": "en"}
    write(tmp_path / "broken.parquet", "not parquet")
    with pytest.raises(DatasetLoadError, match="unreadable Parquet"):
        load_golden_dataset(tmp_path / "broken.parquet")


def test_field_map_and_explicit_format(tmp_path: Path) -> None:
    rows = [{"prompt": "What is M?", "contexts": "M is N.", "gold": "M is N."}]
    path = write(tmp_path / "m.txt", json.dumps(rows))
    loaded = load_golden_dataset(
        path, format="json", field_map={"question": "prompt", "ground_truth": "gold"}
    )
    assert loaded.samples[0].contexts == ("M is N.",)


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("dup.json", json.dumps(RAGAS_ROWS + RAGAS_ROWS), "duplicate sample_id"),
        (
            "nan.json",
            '[{"question": "What?", "contexts": ["a"], "ground_truth": "b",'
            ' "metadata": {"x": NaN}}]',
            "non-finite",
        ),
        (
            "short.json",
            '[{"question": "Hi", "contexts": ["a"], "ground_truth": "b"}]',
            "sample 1: question",
        ),
        ("noctx.json", '[{"question": "What?", "contexts": [], "ground_truth": "b"}]', "contexts"),
        (
            "v2.json",
            json.dumps(
                {
                    "name": "a",
                    "dataset_version": "1.0.0",
                    "schema_version": "2.0.0",
                    "samples": RAGAS_ROWS,
                }
            ),
            "schema_version",
        ),
        (
            "extra.json",
            json.dumps({"name": "a", "owner": "me", "samples": RAGAS_ROWS}),
            "unknown dataset fields",
        ),
        ("scalar.json", "42", "expected a list"),
        ("mixed.json", '[{"question": "What?"}, 3]', "expected a JSON object"),
        ("invalid.json", "{not json", "invalid JSON"),
        ("badname.json", json.dumps({"name": 7, "samples": RAGAS_ROWS}), "name must be a string"),
        (
            "baddesc.json",
            json.dumps({"name": "a", "description": 7, "samples": RAGAS_ROWS}),
            "description must be a string",
        ),
        ("badname2.json", json.dumps({"name": "bad name", "samples": RAGAS_ROWS}), "name"),
        (
            "badmeta.json",
            '[{"question": "What?", "contexts": ["a"], "ground_truth": "b", "metadata": "[1]"}]',
            "metadata: expected an object",
        ),
        (
            "badmeta2.json",
            '[{"question": "What?", "contexts": ["a"], "ground_truth": "b", "metadata": "{oops"}]',
            "metadata: invalid JSON",
        ),
        (
            "badctx.json",
            '[{"question": "What?", "contexts": "[oops", "ground_truth": "b"}]',
            "contexts: invalid JSON",
        ),
        ("lines.jsonl", '{"question": "What?"}\n{oops\n', "line 2: invalid JSON"),
        ("lines2.jsonl", "[1]\n", "line 1: expected a JSON object"),
        (
            "ragged.csv",
            "question,contexts,ground_truth\nWhat is A?,A,B,EXTRA\n",
            "more cells than header",
        ),
        ("x.txt", "[]", "cannot infer the format"),
    ],
)
def test_malformed_inputs(tmp_path: Path, name: str, content: str, message: str) -> None:
    with pytest.raises(DatasetLoadError, match=message):
        load_golden_dataset(write(tmp_path / name, content))


def test_errors_never_echo_row_content(tmp_path: Path) -> None:
    path = write(
        tmp_path / "secret.json",
        '[{"question": "Hi", "contexts": ["SECRET"], "ground_truth": "b"}]',
    )
    with pytest.raises(DatasetLoadError) as error:
        load_golden_dataset(path)
    assert "SECRET" not in str(error.value)
    assert "Hi" not in str(error.value)


def test_header_contradictions_and_metadata_clash(
    tmp_path: Path, reference_dataset: GoldenDataset
) -> None:
    path = write(tmp_path / "doc.json", reference_dataset.to_json())
    with pytest.raises(DatasetLoadError, match="contradicts the file header"):
        load_golden_dataset(path, dataset_version="9.9.9")
    rows = [
        {
            "question": "What?",
            "contexts": ["a"],
            "ground_truth": "b",
            "lang": "x",
            "metadata": {"lang": "y"},
        }
    ]
    with pytest.raises(DatasetLoadError, match="clash with metadata"):
        load_golden_dataset(write(tmp_path / "clash.json", json.dumps(rows)), extra="metadata")


def test_file_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(DatasetLoadError, match="is not a file"):
        load_golden_dataset(tmp_path / "missing.json")
    with pytest.raises(DatasetLoadError, match="is not a file"):
        load_golden_dataset(tmp_path)
    monkeypatch.setattr(dataset_loader, "MAX_DATASET_BYTES", 10)
    with pytest.raises(DatasetLoadError, match="the limit is 10"):
        load_golden_dataset(write(tmp_path / "big.json", json.dumps(RAGAS_ROWS)))


def test_build_dataset_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(DatasetLoadError, match="unknown fields"):
        build_dataset(RAGAS_ROWS, name="x", field_map={"prompt": "q"})
    monkeypatch.setattr(dataset_loader, "MAX_SAMPLES", 1)
    with pytest.raises(DatasetLoadError, match="more than 1 samples"):
        build_dataset(RAGAS_ROWS, name="x")
    with pytest.raises(DatasetLoadError, match="samples"):
        build_dataset([], name="x")


def test_detect_format() -> None:
    assert detect_format(Path("a.NDJSON")) == "jsonl"
    assert detect_format(Path("a.pq")) == "parquet"
