# FVM0 6px Reed-Solomon probe

`FVM0_RS_PROBE` is an isolated physical-channel experiment. It does not modify `FVM0_RAW`, FVM0-T, FOV1, or the file-transfer path.

The default 1920x1080, 6px matrix contains 320x180 = 57,600 bits or 7,200 physical bytes per frame. Each frame carries 28 independent RS(255,239) codewords: 6,692 logical bytes become 7,140 coded bytes, followed by 60 deterministic unprotected reserved bytes. RS efficiency is `239/255`; frame logical efficiency is `6692/7200`, about 92.94%. At 30 FPS the logical probe capacity before outer FEC and protocol overhead is 200,760 bytes/s. This is not real file throughput.

Logical frames contain `F6RS`, version 1, a big-endian uint32 frame index, SHAKE256 deterministic probe bytes, and a final big-endian CRC32. The SHAKE domain is `FVM0_RS_PROBE| || seed:uint64be || frame_index:uint32be`. Reserved bytes use the separate `FVM0_RS_RESERVED|` domain. Encoded codewords are byte-column interleaved as `coded.T.reshape(-1)`. Bits use big-endian packing and a fixed luma threshold of 128.

The decoder reports raw bit/byte errors, reserved-area errors, per-codeword ground-truth raw byte errors, RS API successes/failures, decoder-reported corrections, header/CRC validity, and oracle exact recovery. A codeword API success is not treated as a recovered frame unless the complete logical frame passes validation. Frames that cannot be fully decoded do not receive a fabricated post-FEC BER.

```powershell
python .\scripts\fvm0_rs_encode.py .\runs\fvm0-rs-6px-1200.mp4 --frames 1200 --cell-size 6 --seed 20260809
python .\scripts\fvm0_rs_decode.py .\runs\fvm0-rs-6px-1200.mp4 .\runs\fvm0-rs-6px-1200.manifest.json --output-dir .\runs\fvm0-rs-6px-local
```

Require local exact recovery before manually using an existing platform workflow. Disable platform watermarks, wait for and explicitly download the 1920x1080 rendition, then decode it with the original manifest into a new result directory. Compare raw BER, the codeword raw-byte-error histogram, RS failures, CRC failures, and exact-frame recovery rate.

The ideal platform result has nonzero raw BER but zero RS failures and CRC failures, with every frame recovered exactly. A future outer erasure code may handle residual whole-frame failures, but RaptorQ, packets, synchronization, and real payload transport are deliberately outside this probe.

Dependency: `reedsolo==1.7.0`.
