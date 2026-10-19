import argparse
import time
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from gru_kernel import custom_gru
except ImportError:
    custom_gru = None


def naive_gru(x, w_ih, w_hh, b_ih, b_hh):
    batch, seq_len, _ = x.shape
    h = x.new_zeros(batch, w_hh.shape[1])
    outputs = []
    for t in range(seq_len):
        gi = x[:, t] @ w_ih.T + b_ih
        gh = h @ w_hh.T + b_hh
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h = (1 - z) * n + z * h
        outputs.append(h)
    return torch.stack(outputs, dim=1)


def load_liquidity_gru(path):
    sd = torch.load(path, map_location=DEVICE)
    sd = {k[len("gru."):]: v for k, v in sd.items() if k.startswith("gru.")}
    assert "weight_ih_l1" not in sd, "one layer only"
    gru = nn.GRU(sd["weight_ih_l0"].shape[1], sd["weight_hh_l0"].shape[1], batch_first=True)
    gru.load_state_dict(sd)
    return gru.to(DEVICE).eval()


def params_of(gru):
    return gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0


def time_fn(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000 / iters


def check(name, fn, gru, atol=1e-5):
    x = torch.randn(8, 20, gru.input_size, device=DEVICE)
    with torch.no_grad():
        ref, _ = gru(x)
        try:
            out = fn(x, *params_of(gru))
        except NotImplementedError:
            return False
    err = (out - ref).abs().max().item()
    assert err < atol, f"{name} disagrees with nn.GRU: max abs error {err:.2e}"
    print(f"correctness OK: {name} vs nn.GRU, max abs error {err:.2e}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", help="state_dict of the trained TimeSeriesGRU")
    args = parser.parse_args()

    print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name()})" if DEVICE == "cuda" else ""))
    torch.manual_seed(0)
    if args.weights:
        gru = load_liquidity_gru(args.weights)
        print(f"liquidity GRU: trained weights from {args.weights}")
    else:
        gru = nn.GRU(22, 32, batch_first=True).to(DEVICE).eval()
        print("liquidity GRU: random weights, same shape (pass --weights for the trained model)")

    check("naive", naive_gru, gru)
    ready = custom_gru is not None and check("custom", custom_gru, gru)
    if not ready:
        print("custom kernel: not written yet, skipping that column")

    configs = [(32, 20), (32, 100), (256, 100), (1024, 100)]
    header = f"{'batch':>6} {'seq':>5} | {'naive ms':>9} {'nn.GRU ms':>10}"
    if ready:
        header += f" {'custom ms':>10} {'custom/nn.GRU':>14}"
    print("\n" + header)
    for batch, seq_len in configs:
        x = torch.randn(batch, seq_len, gru.input_size, device=DEVICE)
        with torch.no_grad():
            t_naive = time_fn(lambda: naive_gru(x, *params_of(gru)))
            t_ref = time_fn(lambda: gru(x))
            row = f"{batch:>6} {seq_len:>5} | {t_naive:>9.3f} {t_ref:>10.3f}"
            if ready:
                t_c = time_fn(lambda: custom_gru(x, *params_of(gru)))
                row += f" {t_c:>10.3f} {t_c / t_ref:>13.2f}x"
        print(row)


if __name__ == "__main__":
    main()
