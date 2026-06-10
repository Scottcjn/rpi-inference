# SPDX-License-Identifier: MIT
import importlib.util
import struct
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "gen_test_model.py"
SPEC = importlib.util.spec_from_file_location("gen_test_model", MODULE_PATH)
gen_test_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen_test_model)


def _read_model_header(path):
    data = path.read_bytes()
    return data, struct.unpack_from("<13I76s", data, 0)


def test_write_rpi_header_counts_and_offsets(tmp_path):
    model_path = tmp_path / "test_model.rpi"

    gen_test_model.write_rpi(model_path)

    data, header = _read_model_header(model_path)
    n_routes = gen_test_model.N_CELLS * gen_test_model.N_ROUTES_PER
    n_emits = gen_test_model.N_CELLS * gen_test_model.N_EMITS_PER
    expected_embed_offset = (
        128
        + gen_test_model.N_BANKS * 32
        + gen_test_model.N_CELLS * gen_test_model.CELL_SIZE
        + gen_test_model.N_PERM_BLOCKS * (gen_test_model.RPI_LANES + 8)
        + n_routes * 8
        + n_emits * 8
    )

    assert gen_test_model.CELL_SIZE == struct.calcsize("<IIIIIIII HH")
    assert header[:13] == (
        gen_test_model.RPI_MAGIC,
        gen_test_model.RPI_VERSION,
        gen_test_model.N_CELLS,
        gen_test_model.N_PERM_BLOCKS,
        n_routes,
        n_emits,
        gen_test_model.VOCAB_SIZE,
        gen_test_model.RPI_LANES,
        8,
        32,
        gen_test_model.N_BANKS,
        expected_embed_offset,
        0,
    )
    assert len(data) == expected_embed_offset + gen_test_model.VOCAB_SIZE * 4


def test_write_rpi_bank_descriptors_partition_cells(tmp_path):
    model_path = tmp_path / "test_model.rpi"

    gen_test_model.write_rpi(model_path)

    data = model_path.read_bytes()
    cells_per_bank = gen_test_model.N_CELLS // gen_test_model.N_BANKS
    for bank_id in range(gen_test_model.N_BANKS):
        offset = 128 + bank_id * 32
        bank = struct.unpack_from("<IIIIII8s", data, offset)
        assert bank[:6] == (
            bank_id,
            bank_id * cells_per_bank,
            cells_per_bank,
            bank_id,
            bank_id * 64,
            0xFF,
        )


def test_write_rpi_output_is_deterministic(tmp_path):
    first = tmp_path / "first.rpi"
    second = tmp_path / "second.rpi"

    gen_test_model.write_rpi(first)
    gen_test_model.write_rpi(second)

    assert first.read_bytes() == second.read_bytes()
