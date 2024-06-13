from elsa import Elsa, predict

elsa = Elsa.from_unified()
prompts = elsa.prompts
prompts = prompts.index[::-1]
# prompts = slice(None)
files = slice(None)

# please make sure https://github.com/IDEA-Research/GroundingDINO is installed correctly.
# note: GroundingDINO does not have a setup.py or pyproj.toml, so it does not create a module in your venv
#   you have to just clone it to your directory and let our module access it relative to this directory
predict(
    elsa,
    files=files,
    prompts=prompts,
    outdir='~/Downloads/predict2',
    checkpoint='/home/arstneio/Downloads/gdinot-coco-ft.pth',
    config='/home/arstneio/PycharmProjects/Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py',
    method='batched_without_interpolation',
    force=False,
)
