# SpokenWOZ audio acquisition: five early training dialogues

## Official source and rights

The [SpokenWOZ project](https://spokenwoz.github.io/) links the publisher-associated [Hugging Face training/development audio repository](https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Audio). At pinned revision `5748a6e1b67b26a0e2fca5df061ed7ed2f782eab`, its Hub file inventory contains `.gitattributes` and **one** `audio_5700_train_dev.tar.gz`, exactly **10,341,654,034 bytes**. There are no separately addressable `.wav` repository files. The Hub dataset viewer reports `DatasetGenerationError`; its preview does not constitute a working individual-WAV download endpoint. Do not use an unofficial mirror.

The official site distributes the dataset under **CC BY-NC 4.0** (attribution, noncommercial use); access to the public archive does not itself grant permission for commercial deployment or unrestricted redistribution. Confirm that the intended use fits the license; seek separate permission if it is commercial. Keep recordings under git-ignored `data/raw/` and do not publish them or derived weights without rights review.

## Five selected from the existing ten-dialogue manifest

All five have valid word-interval metadata and populated dialogue-state annotations. Lower overlap counts and early position in the observed archive order guided the selection. Overlaps still require review, and timestamp *metadata* has not been aligned against audio.

| Dialogue ID | Turns | Adjacent overlaps | Turns with populated state slots | Expected member (observed in tar stream) |
| --- | ---: | ---: | ---: | --- |
| MUL0009 | 36 | 0 | 16 | `audio_5700_train_dev/MUL0009.wav` |
| MUL0008 | 60 | 4 | 29 | `audio_5700_train_dev/MUL0008.wav` |
| MUL0013 | 54 | 4 | 19 | `audio_5700_train_dev/MUL0013.wav` |
| MUL0006 | 24 | 4 | 12 | `audio_5700_train_dev/MUL0006.wav` |
| MUL0003 | 42 | 8 | 19 | `audio_5700_train_dev/MUL0003.wav` |

Counts come from `data/processed/spokenwoz/sample_10_turns.jsonl`, except the populated-state counts, which inspect nonblank `metadata.*.semi` values in the same ten original sample dialogues. This is a limited selection heuristic, not a validated alignment ranking.

## Verified bounded access (no WAVs saved)

The pinned Hub `resolve` URL accepted an HTTP `Range: bytes=0-50331647` request and returned **206 Partial Content**, `Content-Range: bytes 0-50331647/10341654034`. The returned prefix began with gzip magic bytes `1f8b`; Python's `tarfile.open(fileobj=response, mode="r|gz")` listed members in sequence. A bounded probe consumed **29,102,080 compressed bytes** through `MUL0013.wav` and read that member's complete **6,954,924-byte** uncompressed body, then closed the response **without saving any WAV**. Earlier members included all four other selected WAVs. Together the five tar-member WAV sizes are **34,067,164 bytes (~32.49 MiB)**. This establishes that these five can be selectively extracted from the **beginning** of the official stream without transferring the full 10.34 GB archive; extraction itself was not performed in this task.

This is *sequential prefix streaming*, not random access to members. Gzip compression prevents jumping to a tar offset with an HTTP byte range; `tarfile` must decompress skipped members from the beginning. The 48 MiB range is a hard network upper bound for this attempt; if the server stops honoring ranges, a script must refuse the response instead of silently downloading the whole archive. For these particular five early members, the bounded probe confirms the prefix is sufficient. A member late in the archive may require most or all compressed bytes. Hugging Face `hf download` downloads the repository file as an archive; `datasets` streaming can process tar records sequentially but does not create independent per-WAV Hub files. The broken viewer is not a reliable access route.

## Recommended next action — bounded selective extraction, *not run yet*

Once noncommercial-use terms and local storage are acceptable, run **from the repository root**. This reads at most the first 48 MiB of the pinned official archive, refuses a non-206 response, writes only the five selected WAVs to the ignored raw-data directory, and exits as soon as all are complete. It checks each extracted byte count against its tar member size; it does not assert waveform/transcript alignment.

```bash
python3 - <<'PY'
import shutil
import tarfile
import urllib.request
from pathlib import Path

revision = '5748a6e1b67b26a0e2fca5df061ed7ed2f782eab'
url = ('https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Audio/'
       f'resolve/{revision}/audio_5700_train_dev.tar.gz')
selected = {'MUL0003.wav', 'MUL0006.wav', 'MUL0008.wav',
            'MUL0009.wav', 'MUL0013.wav'}
output = Path('data/raw/spokenwoz/audio_5700_train_dev')
output.mkdir(parents=True, exist_ok=True)
cap = 48 * 1024 * 1024
request = urllib.request.Request(url, headers={'Range': f'bytes=0-{cap-1}'})
with urllib.request.urlopen(request, timeout=120) as response:
    if response.status != 206:
        raise RuntimeError('Range not honored; refusing unbounded download')
    with tarfile.open(fileobj=response, mode='r|gz') as archive:
        found = set()
        for member in archive:
            filename = member.name.rsplit('/', 1)[-1]
            if (filename not in selected or not member.isfile()
                    or member.name != f'audio_5700_train_dev/{filename}'):
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f'Cannot read {filename}')
            partial = output / f'{filename}.partial'
            try:
                with source, partial.open('wb') as target:
                    shutil.copyfileobj(source, target)
                if partial.stat().st_size != member.size:
                    raise RuntimeError(f'Incomplete {filename}')
                partial.replace(output / filename)
            finally:
                partial.unlink(missing_ok=True)
            found.add(filename)
            if found == selected:
                break
if found != selected:
    raise RuntimeError(f'Missing WAV members: {selected - found}')
print('Extracted:', sorted(found))
PY
```

Reserve at least **~50 MiB of free space** for the five outputs plus temporary `.partial` files and overhead; no 10.34 GB archive copy is needed for this bounded route. Check actual free space before proceeding. After extraction, inspect every WAV with `soundfile.info`, check its duration against raw word-offset maxima and verify whether raw units are milliseconds, then listen to short selected turns and review overlaps/role mapping. The existing turn manifest's `audio_validation_status` must not be treated as verified merely because filenames exist; updating it requires a separate validation task.

If bounded access fails or different later IDs are required, use `hf download ssz1111/SpokenWOZ-Train-Audio audio_5700_train_dev.tar.gz --repo-type dataset --revision 5748a6e1b67b26a0e2fca5df061ed7ed2f782eab --local-dir data/raw/spokenwoz` **only after explicit approval**; that alternative stores the entire 10,341,654,034-byte compressed archive and needs additional extraction space. It was not started here.
