"""Create a contact sheet from slides exported by PowerPoint."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.tools/presentation'))
from PIL import Image, ImageDraw

folder = ROOT / 'latex/slides'
files = sorted((folder / 'preview').glob('*.PNG'),
               key=lambda p: int(''.join(c for c in p.stem if c.isdigit())))
if not files:
    raise SystemExit('Export slide PNGs from PowerPoint into latex/slides/preview first.')
sheet = Image.new('RGB', (1600, ((len(files)+3)//4)*245), '#e5e5e5')
draw = ImageDraw.Draw(sheet)
for i, path in enumerate(files):
    with Image.open(path) as image:
        image.thumbnail((390, 219))
        x, y = (i % 4)*400+5, (i//4)*245+22
        sheet.paste(image, (x, y))
        draw.text((x, y-17), f'Slide {i+1:02d}', fill='black')
sheet.save(folder / 'contact_sheet.png')
print(f'Created preview for {len(files)} slides.')
