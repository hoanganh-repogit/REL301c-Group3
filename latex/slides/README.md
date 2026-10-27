# Snake DQN research presentation

Main file: `Snake_DQN_Research_Presentation.pptx`.
Quick-read version: `Snake_DQN_Research_Presentation.pdf`.
Overview of all slides: `contact_sheet.png`.

## Content and usage

- Slides are in English; the Vietnamese talk track is in the **Speaker Notes** of
  each slide.
- The talk track is also in `speaker_notes_vi.md` for rehearsal.
- Slides 1–19: main talk, about 12–15 minutes.
- Slides 20–21: detailed table and backup questions for the defense.
- In PowerPoint, use **Slide Show → Presenter View** to see the notes while
  presenting.
- Fill in the authors, affiliation, and supervisor on the first slide before
  submission.

## Style

Built from the template `latex/slides/template.pptx`.
It keeps the 16:9 size, the grey-green background `#788B73`, the black frame,
the pixel decorations on the first and last slides, and the Pixellance title
font embedded in the template.
Scientific content uses Aptos so that tables, formulas, and numbers stay
readable. Text, tables, and success-rate bars remain editable in PowerPoint; the
analysis charts are images from the report pipeline.

## Data

The content follows `../main.tex`, `../../configs/experiments/dqn_final_mastery.yaml`,
`../../results/cross_level_evaluation/summary.csv`, and the charts in `../generated/`.
It presents only the Dueling Double DQN trained on Level 4, with the best
checkpoint, the final checkpoint, and the random baseline.

## Regenerating

From the repository root. The packages in `.tools/presentation` are built for
Python 3.14, so use `py -3.14`:

```powershell
py -3.14 -m pip install --target .tools/presentation python-pptx matplotlib numpy pyyaml
py -3.14 latex/scripts/generate_report_assets.py
py -3.14 latex/scripts/generate_presentation.py
```

Open the PPTX in PowerPoint and export it to PDF. If the slide PNGs are exported
into `latex/slides/preview`, the overview image can be created with:

```powershell
py -3.14 latex/scripts/preview_presentation.py
```

The scripts overwrite their output files. The original template is never
modified.

**Font note:** on a machine without the Pixellance font installed, PowerPoint
exports (PDF or PNG) fall back to a plain font for the slide titles, even though
the font is embedded in the PPTX. Install Pixellance in Windows before
exporting.
