from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path

from elsa import *

indir = '/home/arstneio/Downloads/top10/'
indir = Path(indir)
outdir = '/home/arstneio/Downloads/top10images/'
outdir = Path(outdir)
force = False
inpaths: list[Path] = list(indir.rglob('*.parquet'))

elsa = Elsa.from_unified()
files = elsa.files.file

with ThreadPoolExecutor() as threads:
    futures = []
    i = 0
    total = len(inpaths) * len(files)

    for inpath in inpaths:
        logits = Logits.from_file(inpath, elsa=elsa)
        for file in files:
            print(f'{i}/{total}')


            i += 1
            p = inpath.relative_to(indir)
            parts = p.parts

            new_path = Path(*parts)
            outpath = (
                    outdir
                    / file
                    / new_path
                    .with_suffix('.png')
            )
            if outdir.exists():
                continue
            if (
                    not force
                    and outpath.exists()
            ):
                continue

            outpath.parent.mkdir(parents=True, exist_ok=True)
            try:
                image = logits.view_top(file=file, background='white',show_filename=False)
            except Exception:
                continue
            future = threads.submit(image.save, outpath)
            futures.append(future)

    for future in as_completed(futures):
        future.result()



