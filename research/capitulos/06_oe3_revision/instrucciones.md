# Three tasks

Do task 1 first, then task 2. Task 3 is optional.
While you work, don't open your original annotation tables for these recordings.

## Task 1 · Timed annotation (`1_timed_session`)

{n_folders} folders with about {folder_minutes} minutes of audio each. Do them in order: 1, 2, 3, 4.

- **`manual` folders:** annotate each recording as you normally do.
- **`assisted` folders:** open the model's table for each recording (same name, ending in
  `.model.txt`). Delete the wrong boxes, fix the others and add the calls it missed.
- In both, save one selection table per recording in the same folder, with Raven's default name.
  Work at your normal pace and quality.
- In `times.csv`, write the clock time when you start and finish each folder (for example
  `14:05`) and the minutes of any break.

## Task 2 · Check the model's boxes (`2_check_boxes`, about {check_hours} hours)

`boxes.csv` has {n_boxes} boxes. Go from the top down without skipping rows.

For each row:

1. In Raven, open the recording named in `file` (in `audio/`) and its table (same name, ending in
   `.boxes.txt`). Find the box by its `ID`.
2. Write one word in `answer`:

| Answer | When | `comment` (optional) |
|---|---|---|
| `correct` | It is the call in `model_says` and the box fits it. | |
| `other` | It is a primate call, but of another species or call type. | which call |
| `box` | It is the call in `model_says`, but the box is too big, cut short or covers several calls. | |
| `no` | There is no primate call in the box. | what it is: insect, bird, rain… |
| `?` | You can't tell. | |

If both the call and the box are wrong, write `other`. Some boxes are right and some are wrong:
judge each one on its own.

## Task 3 · Dataset quality (`3_dataset_quality`, optional)

`boxes.csv` has {n_findings} boxes from the rest of the dataset. Answer as in task 2, judging the
call in `box_says`.

- Audio: in your own archive, at `folder`/`file`.
- Table: in `tables/<folder>/`, same name, ending in `.quality.txt`.

Go from the top down without skipping rows. Stop whenever you like.

## Send back

- The three folders, with both `boxes.csv`, `times.csv` and the tables you saved.
- A short note: who did it, their years of experience, and whether they annotated any of these
  recordings before.
