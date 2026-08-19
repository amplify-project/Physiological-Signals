# Engagement Action Inventory — Kinetics-700

> Purpose: reference for the retrain discussion. Splits the source action catalog into
> **actions we currently use** (the 86-class subset extracted under
> `data/processed/features_kinetics700/`) and **actions we do not yet use** (the remaining
> 614 Kinetics-700 classes). Each currently-used action carries a suitability flag for the
> **young-families concert audience** context.

- **Source dataset:** Kinetics-700 — 700 action classes total.
- **Currently used:** 86 fine classes → mapped to a binary head (engagement / disengagement) via
  [server/labels/hierarchical_labels.json](server/labels/hierarchical_labels.json).
- **Not yet used:** 614 classes (700 − 86).
- **Current model:** `models/action_transformer_12gpus_binary_v2_cleaned` (all 86 fed as features).
- **Young-families variant:** `models/action_transformer_young_families_v0` — same 86, only `staring`
  re-labelled to engagement and engagement loss up-weighted 1.5×.

## Model contexts

Engagement is context-dependent, so we keep **separate models per viewing context** rather than one
universal scorer. Two contexts are in scope:

| Context | Audience | Best-suited model | Why |
|---------|----------|-------------------|-----|
| **Webcam / online-learning (higher-arousal, close, seated)** | One or few people close to a webcam, often looking straight at the camera | `action_transformer_young_families_v0` (the original arousal/action model; engagement F1 0.74) | Proximity + direct-to-camera gaze + expressive action carry the signal; arousal correlates with engagement here. |
| **Physical family concert (docile, seated, at distance)** | A room of families watching live performers, faces small and off-axis | `action_transformer_family_concert_v1` (new, attention-first — this document) | Low arousal ≠ disengagement; attentive stillness dominates; distance makes proximity/expression unreliable. |

The original model despite its `young_families` name is really the **webcam / higher-arousal** model;
it stays **frozen** and may still be tweaked/retrained for that context. The new attention-first model
below is the one for the physical seated family-concert context. A future app `--context` flag will
select the right model + label file (see runbook §5).

## Engagement philosophy (attention-first)

This model targets a **docile, mostly-seated family audience** watching performers, so scoring is
*attention-first*, not arousal-based:

- **Presume engaged.** Start from a high baseline and *subtract* on clear anti-engagement cues.
- **Low arousal is not disengagement.** A quiet, still, seated person watching the stage is the
  **most engaged** state — not a neutral or passive one. Quiet **staring / attentive stillness** is
  the single strongest positive signal.
- **Disengagement = distraction, distress, restlessness or boredom** — phone use, looking away,
  fidgeting / excess movement ("kids moving around a lot"), yawning, sleeping, crying, a child falling.
- **Neutral / involuntary / consumption actions are excluded** (eating, drinking, coughing, sneezing,
  burping, drooling) — they carry no engagement signal and only add noise.
- **The strongest cues are not Kinetics "actions".** Attentive stillness, looking-away and excess
  movement are better captured by the **gaze / orientation / motion-energy feature layer** than by
  action classes; the action head is used mainly as a **distraction / distress detector** on top of the
  attentive-presence baseline. (See ROADMAP → distance-robust fusion fallback.)

> **The action head is a disengagement detector, by design.** Because we *presume engaged and only
> detect disengagement*, the training set is deliberately imbalanced toward disengagement and the head
> is judged on its **disengagement** metrics, not engagement recall (the engaged baseline is supplied by
> the gaze layer). The trained `family_concert_v1` (weight 1.5) gives disengagement **recall 0.93** /
> **precision 0.76** — it catches most disengagement with a modest false-alarm rate. The lever for
> fewer false alarms (an engaged person wrongly flagged) is a *lower* engagement weight or a higher
> disengagement threshold — **not** rebalancing toward engagement.

### Suitability flag legend (for a docile, mostly-seated family audience watching performers)

| Flag | Meaning |
|------|---------|
| ✅ KEEP | Relevant to an audience **and** reasonably detectable from pose keypoints |
| 🔁 RE-LABEL | Occurs, but its engagement/disengagement meaning should flip for this context |
| ⚖️ RE-WEIGHT | Relevant but weak/undetectable from pose alone → keep at reduced confidence |
| ❌ DROP | Irrelevant, inappropriate, or a **performer** (not audience) action for this context |

Performance notes (F1) are from [docs/action_transformer_results_nov2025.md](docs/action_transformer_results_nov2025.md).
`F1=0.00` marks classes the current model completely fails to detect.

---

## 1. Actions we CURRENTLY use (86)

### 1a. Engagement classes (26)

| idx | fine class | macro | flag | note |
|----:|------------|-------|:----:|------|
| 0 | applauding | applause | ✅ KEEP | core audience cue |
| 9 | clapping | applause | ✅ KEEP | core audience cue |
| 2 | belly dancing | dancing | ❌ DROP | top-F1 (0.39) but won't occur in a seated family audience |
| 4 | breakdancing | dancing | ❌ DROP | inappropriate for context |
| 11 | country line dancing | dancing | ❌ DROP | F1 0.40 but audience won't do this |
| 14 | dancing ballet | dancing | ❌ DROP | performer/stage action |
| 15 | dancing charleston | dancing | ❌ DROP | won't occur |
| 16 | dancing gangnam style | dancing | ❌ DROP | won't occur |
| 17 | dancing macarena | dancing | ⚖️ RE-WEIGHT | group/kids dance possible, low prior <kids moving around a lot is dissengagment>| 
| 33 | jumpstyle dancing | dancing | ❌ DROP | too energetic for context |
| 36 | mosh pit dancing | dancing | ❌ DROP | inappropriate for family audience |
| 56 | robot dancing | dancing | ⚖️ RE-WEIGHT | playful kids’ movement, low prior <kids moving around a lot is dissengagment>|
| 58 | salsa dancing | dancing | ❌ DROP | won't occur |
| 60 | shoot dance | dancing | ⚖️ RE-WEIGHT | kids may do this, low prior  <kids moving around a lot is dissengagment> |
| 68 | square dancing | dancing | ❌ DROP | won't occur |
| 73 | swing dancing | dancing | ❌ DROP | won't occur |
| 75 | tango dancing | dancing | ❌ DROP | won't occur |
| 76 | tap dancing | dancing | ❌ DROP | performer action |
| 7 | cheerleading | cheering | ❌ DROP | performer-style, not this audience |
| 32 | headbanging | cheering | ❌ DROP | inappropriate for family context |
| 72 | surfing crowd | cheering | ❌ DROP | won't/should not occur |
| 61 | singing | singing | ⚖️ RE-WEIGHT | sing-along is relevant but **F1=0.00** (pose can't see mouth) |
| 31 | gospel singing in church | singing | ⚖️ RE-WEIGHT | proxy for sing-along; undetectable from pose |
| 37–52 | playing accordion…violin (16 instruments) | playing_instrument | ❌ DROP | **performer** actions; audience does not play instruments (performers are already excluded from scoring) |
| 55 | recording music | recording | ❌ DROP | **F1=0.00**, device/tech action |
| 80 | using megaphone | recording | ❌ DROP | **F1=0.00**, performer/staff action |

> The 16 instrument classes (idx 37–52) are collapsed into one row: `playing accordion, playing
> bagpipes, playing bass guitar, playing cello, playing clarinet, playing cymbals, playing drums,
> playing guitar, playing harmonica, playing keyboard, playing piano, playing saxophone, playing
> trombone, playing trumpet, playing ukulele, playing violin`. Cello/violin were the model's best
> instrument detectors (F1 ≈ 0.44/0.42) but they describe the **performers**, not the audience.

### 1b. Disengagement classes (60)

| idx | fine class | macro | flag | note |
|----:|------------|-------|:----:|------|
| 35 | looking at phone | phone_distraction | ⚖️ RE-WEIGHT | genuine cue but **F1=0.00** (looks like reading/staring) |
| 78 | texting | phone_distraction | ✅ KEEP | genuine disengagement |
| 74 | talking on cell phone | phone_distraction | ✅ KEEP | genuine disengagement |
| 34 | listening with headphones | phone_distraction | ⚖️ RE-WEIGHT | uncommon in this audience |
| 63 | sleeping | passive | ✅ KEEP | child/adult asleep = disengaged |
| 85 | yawning | passive | ✅ KEEP | boredom cue |
| 69 | staring | passive | 🔁 RE-LABEL | already flipped to **engagement** in young-families model (sustained attention) |
| 83 | watching tv | passive | ❌ DROP | no TV at a live concert |
| 53 | reading book | passive | ⚖️ RE-WEIGHT | possible (programme) but ambiguous |
| 54 | reading newspaper | passive | ❌ DROP | **F1=0.00**, won't occur |
| 81 | waiting in line | passive | ❌ DROP | not an in-audience action |
| 30 | fidgeting | fidgeting | ✅ KEEP | kids fidget; useful restlessness cue |
| 79 | twiddling fingers | fidgeting | ⚖️ RE-WEIGHT | subtle hand motion, weak |
| 20 | drumming fingers | fidgeting | ⚖️ RE-WEIGHT | **F1=0.00**, too subtle for pose |
| 77 | tapping pen | fidgeting | ⚖️ RE-WEIGHT | subtle, low prior |
| 57 | rolling eyes | negative_body_language | ❌ DROP | face-based, not visible from pose |
| 59 | shaking head | negative_body_language | ⚖️ RE-WEIGHT | detectable but ambiguous |
| 12 | crossing eyes | negative_body_language | ❌ DROP | **F1=0.00**, face-based |
| 1 | arguing | negative_body_language | ⚖️ RE-WEIGHT | possible but rare in audience |
| 84 | winking | negative_body_language | ❌ DROP | **F1=0.00**, face-based |
| 64 | smoking | smoking | ❌ DROP | won't occur at a family concert |
| 65 | smoking hookah | smoking | ❌ DROP | won't occur |
| 66 | smoking pipe | smoking | ❌ DROP | won't occur |
| 6 | checking watch | checking_time | ✅ KEEP | boredom / time-checking cue |
| 21 | eating burger | eating_drinking | 🔁 RE-LABEL | **F1=0.00**; eating at a family event is **neutral**, not disengaged |
| 22 | eating chips | eating_drinking | 🔁 RE-LABEL | neutral snacking |
| 23 | eating doughnuts | eating_drinking | 🔁 RE-LABEL | neutral snacking |
| 24 | eating hotdog | eating_drinking | 🔁 RE-LABEL | neutral snacking |
| 25 | eating ice cream | eating_drinking | 🔁 RE-LABEL | neutral snacking |
| 26 | eating nachos | eating_drinking | 🔁 RE-LABEL | **F1=0.00**; neutral snacking < i think we can drop eating (not neccisarily dissengagment) >|
| 27 | eating spaghetti | eating_drinking | ❌ DROP | not a concert food; arm-to-mouth dup |
| 18 | drinking shots | eating_drinking | ❌ DROP | inappropriate for family context |
| 62 | sipping cup | eating_drinking | 🔁 RE-LABEL | neutral drinking <drop> |
| 82 | waking up | tired_uncomfortable | ✅ KEEP | disengagement cue |
| 70 | stretching arm | tired_uncomfortable | ⚖️ RE-WEIGHT | ambiguous (could be reaching for child) |
| 71 | stretching leg | tired_uncomfortable | ⚖️ RE-WEIGHT | best-F1 class (0.50) but weak engagement signal <looking at less = dissengaged (we have seen this in smoke test video)> |
| 10 | coughing | tired_uncomfortable | 🔁 RE-LABEL | **F1=0.00**; involuntary/neutral, not disengagement  <drop>|
| 67 | sneezing | tired_uncomfortable | 🔁 RE-LABEL | involuntary/neutral <drop> |
| 3 | blowing nose | tired_uncomfortable | 🔁 RE-LABEL | involuntary/neutral <drop> |
| 13 | crying | negative_reactions | 🔁 RE-LABEL | **F1=0.00**; a crying **child** is distress, handle separately, not simple disengagement <good dissengagement> |
| 5 | burping | negative_reactions | 🔁 RE-LABEL | involuntary/neutral (esp. babies)  <drop> |
| 19 | drooling | negative_reactions | 🔁 RE-LABEL | baby behaviour, neutral <drop> |
| 8 | chewing gum | negative_reactions | ⚖️ RE-WEIGHT | subtle, near-F1 0  <drop>|
| 29 | falling off chair | negative_reactions | ⚖️ RE-WEIGHT | rare, but a real distress/disengage event <good, we have seen kids falling in smoke test video which causes a lot of distraction>|
| 28 | falling off bike | negative_reactions | ❌ DROP | won't occur indoors/seated |

**Currently-used summary:** of the 26 "engagement" classes, most are performer/dance-style actions
that won't occur in a seated family audience; the durable audience-positive cues are essentially
**applauding, clapping** (and, if detectable, **singing/sing-along**). On the disengagement side the
strong keepers are **texting, talking on cell phone, sleeping, yawning, fidgeting, checking watch**;
a large block (eating, coughing/sneezing, burping/drooling, smoking, face-based cues) is either
**neutral, involuntary, undetectable from pose, or won't occur** and is a prime candidate to drop or
re-label before retraining.

<smoke test video shows tapping hands on lap during sing along is engaged>
---

## 2. Revised taxonomy — retrain target (`family_concert_v1`)

Applying the philosophy above and the inline review decisions, the retrain collapses the 86-class set
to a small, defensible action head plus a feature layer. Trainer-ready, machine-readable version:
[server/labels/hierarchical_labels_family_concert.json](server/labels/hierarchical_labels_family_concert.json)
(5 engaged + 9 disengaged kept/added from Kinetics-700, 2 of the engaged — `karaoke`, `laughing` —
still to extract, 74 dropped as neutral). This is a **separate context-specific model**: the original
86-class models are left frozen for the online-learning context (see runbook §5).

### Engaged (attention & calm participation)
| class | source | note |
|-------|--------|------|
| staring | K700 #69 (flipped) | **primary** cue — attentive stillness |
| applauding | K700 #0 | clap |
| clapping | K700 #9 | clap |
| karaoke | *to extract* | sing-along proxy (better-posed than `singing`) |
| laughing | *to extract (optional)* | positive reaction |

### Disengaged (distraction / distress / restlessness / boredom)
| class | source | note |
|-------|--------|------|
| looking at phone | K700 #35 | weak (F1=0.00) but genuine |
| texting | K700 #78 | distraction |
| talking on cell phone | K700 #74 | distraction |
| fidgeting | K700 #30 | restlessness |
| checking watch | K700 #6 | boredom / time-check |
| yawning | K700 #85 | boredom |
| sleeping | K700 #63 | unambiguous — the one low-arousal exception |
| crying | K700 #13 | child distress (also feeds distress channel) |
| falling off chair | K700 #29 | distraction event (seen in smoke test) |

### Feature-layer signals (not action classes)
| signal | polarity | source |
|--------|:--------:|--------|
| attentive stillness / gaze on stage | + | gaze + orientation |
| looking away / reduced attention | − | gaze |
| excess movement / restlessness ("kids moving a lot") | − | motion-energy / synchrony (replaces dance classes) |
| tapping hands on lap to rhythm | + | in-domain data / rhythm cue |

### Dropped as neutral (excluded from training)
All remaining currently-used classes: the 16 instrument (performer) classes, all dance styles,
cheerleading / headbanging / crowd-surfing, singing / gospel (pose-undetectable, replaced by karaoke),
recording music / megaphone, all eating & drinking, coughing / sneezing / blowing nose,
burping / drooling / chewing gum, smoking, watching tv / reading / waiting in line, headphones,
stretching, waking up, twiddling / drumming fingers / tapping pen, rolling / crossing eyes, winking,
shaking head, arguing, falling off bike. See the JSON `dropped_neutral` block for the full list with
reason tags.

---

## 3. High-value candidates from the UNUSED set (context-relevant shortlist)

These are drawn from the 614 unused classes and are the most plausible additions for a
young-families concert audience. The decisions of record are in §2; this section keeps the wider
rationale. Full list follows in §4.

### Positive / engagement candidates
| candidate | why it fits |
|-----------|-------------|
| laughing | strong, visible positive audience reaction <ok maybe> |
| celebrating | positive crowd reaction <no>|
| high fiving | shared positive moment (parent–child) <audience will be docile we cannot penalise low arousal as dissengagement. audiences will be seated, might clap along at most. quiet staring is the highest indicator of engagement> |
| pumping fist | enthusiasm cue |
| waving hand | waving to performers / participation |
| shouting | cheering-style vocal reaction (pose: open posture) |
| answering questions | interactive engagement (call-and-response) |
| finger snapping | rhythm participation |
| marching | movement participation (kids) |
| karaoke | sing-along proxy (better-posed than `singing`) <good> |
| carrying baby | parent attentive-with-child (engaged-present) |
| hugging baby | parent–child affection (engaged-present) |
| moving baby / moving child | active parenting, attentive presence |
| tickling | playful parent–child interaction |
| kissing | family affection (neutral–positive) |
| hugging (not baby) | family affection (neutral–positive) |
| sticking tongue out | playful child engagement |
| raising eyebrows | reaction/surprise cue |

### Negative / disengagement candidates
| candidate | why it fits |
|-----------|-------------|
| throwing tantrum | child disengagement / distress |
| sleeping *(already used)* | — |
| checking tires / using atm etc. | — (none beyond phone/sleep add much) |

> Note: the genuinely useful **disengagement** signals for this context are already largely covered
> by the current phone/sleep/yawn/fidget set. The bigger opportunity is adding **family-positive and
> parent–child** actions above, which Kinetics-700 does contain but we have not extracted.

> Caveat: several ideal cues (gentle **swaying/bouncing with a child**, **smiling**, **nodding along**,
> **attentive stillness**) are **not discrete Kinetics-700 classes** and would remain rule/feature-layer
> signals (gaze, orientation, motion energy), consistent with the ROADMAP fallback work.

---

## 4. Actions we do NOT yet use (614)

Complete alphabetical enumeration of every Kinetics-700 class **not** in the current 86.
Context-relevant candidates from §2 are marked with ★.

abseiling · acting in play · adjusting glasses · air drumming · alligator wrestling · answering questions ★ · applying cream · archaeological excavation · archery · arm wrestling · arranging flowers · arresting · assembling bicycle · assembling computer · attending conference · auctioning · baby waking up · backflip (human) · baking cookies · bandaging · barbequing · bartending · base jumping · bathing dog · battle rope training · beatboxing · bee keeping · being excited · being in zero gravity · bench pressing · bending back · bending metal · biking through snow · blasting sand · blending fruit · blowdrying hair · blowing bubble gum · blowing glass · blowing leaves · blowing out candles · bobsledding · bodysurfing · bookbinding · bottling · bouncing ball (not juggling) · bouncing on bouncy castle · bouncing on trampoline · bowling · braiding hair · breading or breadcrumbing · breaking boards · breaking glass · breathing fire · brush painting · brushing floor · brushing hair · brushing teeth · building cabinet · building lego · building sandcastle · building shed · bulldozing · bungee jumping · busking · calculating · calligraphy · canoeing or kayaking · capoeira · capsizing · card stacking · card throwing · carrying baby ★ · carrying weight · cartwheeling · carving ice · carving marble · carving pumpkin · carving wood with a knife · casting fishing line · catching fish · catching or throwing baseball · catching or throwing frisbee · catching or throwing softball · celebrating ★ · changing gear in car · changing oil · changing wheel (not on bike) · chasing · checking tires · chiseling stone · chiseling wood · chopping meat · chopping wood · clam digging · clay pottery making · clean and jerk · cleaning gutters · cleaning pool · cleaning shoes · cleaning toilet · cleaning windows · climbing a rope · climbing ladder · climbing tree · closing door · coloring in · combing hair · contact juggling · contorting · cooking chicken · cooking egg · cooking on campfire · cooking sausages (not on barbeque) · cooking scallops · cosplaying · counting money · cracking back · cracking knuckles · cracking neck · crawling baby · crocheting · crossing river · cumbia · curling (sport) · curling eyelashes · curling hair · cutting apple · cutting cake · cutting nails · cutting orange · cutting pineapple · cutting watermelon · deadlifting · dealing cards · decorating the christmas tree · decoupage · delivering mail · digging · dining · directing traffic · disc golfing · diving cliff · docking boat · dodgeball · doing aerobics · doing jigsaw puzzle · doing laundry · doing nails · doing sudoku · drawing · dribbling basketball · driving car · driving tractor · drop kicking · dumpster diving · dunking basketball · dyeing eyebrows · dyeing hair · eating cake · eating carrots · eating watermelon · egg hunting · embroidering · entering church · exercising arm · exercising with an exercise ball · extinguishing fire · faceplanting · feeding birds · feeding fish · feeding goats · fencing (sport) · filling cake · filling eyebrows · finger snapping ★ · fixing bicycle · fixing hair · flint knapping · flipping bottle · flipping pancake · fly tying · flying kite · folding clothes · folding napkins · folding paper · front raises · frying vegetables · gargling · geocaching · getting a haircut · getting a piercing · getting a tattoo · giving or receiving award · gold panning · golf chipping · golf driving · golf putting · grinding meat · grooming cat · grooming dog · grooming horse · gymnastics tumbling · hammer throw · hand washing clothes · head stand · headbutting · helmet diving · herding cattle · high fiving ★ · high jump · high kick · historical reenactment · hitting baseball · hockey stop · holding snake · home roasting coffee · hopscotch · hoverboarding · huddling · hugging (not baby) ★ · hugging baby ★ · hula hooping · hurdling · hurling (sport) · ice climbing · ice fishing · ice skating · ice swimming · inflating balloons · installing carpet · ironing · ironing hair · javelin throw · jaywalking · jetskiing · jogging · juggling balls · juggling fire · juggling soccer ball · jumping bicycle · jumping into pool · jumping jacks · jumping sofa · karaoke ★ · kicking field goal · kicking soccer ball · kissing ★ · kitesurfing · knitting · krumping · land sailing · laughing ★ · lawn mower racing · laying bricks · laying concrete · laying decking · laying stone · laying tiles · leatherworking · letting go of balloon · licking · lifting hat · lighting candle · lighting fire · lock picking · long jump · longboarding · looking in mirror · luge · lunge · making a cake · making a sandwich · making balloon shapes · making bubbles · making cheese · making horseshoes · making jewelry · making latte art · making paper aeroplanes · making pizza · making slime · making snowman · making sushi · making tea · making the bed · marching ★ · marriage proposal · massaging back · massaging feet · massaging legs · massaging neck · massaging person's head · metal detecting · milking cow · milking goat · mixing colours · moon walking · mopping floor · motorcycling · mountain climber (exercise) · moving baby ★ · moving child ★ · moving furniture · mowing lawn · mushroom foraging · needle felting · news anchoring · opening bottle (not wine) · opening coconuts · opening door · opening present · opening refrigerator · opening wine bottle · packing · paragliding · parasailing · parkour · passing American football (in game) · passing American football (not in game) · passing soccer ball · peeling apples · peeling banana · peeling potatoes · person collecting garbage · petting animal (not cat) · petting cat · petting horse · photobombing · photocopying · picking apples · picking blueberries · pillow fight · pinching · pirouetting · planing wood · planting trees · plastering · playing american football · playing badminton · playing basketball · playing beer pong · playing billiards · playing blackjack · playing cards · playing checkers · playing chess · playing controller · playing cricket · playing darts · playing didgeridoo · playing dominoes · playing field hockey · playing flute · playing gong · playing hand clapping games · playing harp · playing ice hockey · playing kickball · playing laser tag · playing lute · playing mahjong · playing maracas · playing marbles · playing monopoly · playing netball · playing nose flute · playing oboe · playing ocarina · playing organ · playing paintball · playing pan pipes · playing piccolo · playing pinball · playing ping pong · playing poker · playing polo · playing recorder · playing road hockey · playing rounders · playing rubiks cube · playing scrabble · playing shuffleboard · playing slot machine · playing squash or racquetball · playing tennis · playing volleyball · playing with trains · playing xylophone · poaching eggs · poking bellybutton · pole vault · polishing furniture · polishing metal · popping balloons · pouring beer · pouring milk · pouring wine · preparing salad · presenting weather forecast · pretending to be a statue · pull ups · pulling espresso shot · pulling rope (game) · pumping fist ★ · pumping gas · punching bag · punching person (boxing) · push up · pushing car · pushing cart · pushing wheelbarrow · pushing wheelchair · putting in contact lenses · putting on eyeliner · putting on foundation · putting on lipstick · putting on mascara · putting on sari · putting on shoes · putting wallpaper on wall · raising eyebrows ★ · repairing puncture · riding a bike · riding camel · riding elephant · riding mechanical bull · riding mule · riding or walking with horse · riding scooter · riding snow blower · riding unicycle · ripping paper · roasting marshmallows · roasting pig · rock climbing · rock scissors paper · roller skating · rolling pastry · rope pushdown · running on treadmill · sailing · saluting · sanding floor · sanding wood · sausage making · sawing wood · scrambling eggs · scrapbooking · scrubbing face · scuba diving · seasoning food · separating eggs · setting table · sewing · shaking hands · shaping bread dough · sharpening knives · sharpening pencil · shaving head · shaving legs · shearing sheep · shining flashlight · shining shoes · shooting basketball · shooting goal (soccer) · shooting off fireworks · shopping · shot put · shouting ★ · shoveling snow · shredding paper · shucking oysters · shuffling cards · shuffling feet · side kick · sieving · sign language interpreting · silent disco · situp · skateboarding · ski ballet · ski jumping · skiing crosscountry · skiing mono · skiing slalom · skipping rope · skipping stone · skydiving · slacklining · slapping · sled dog racing · slicing onion · smashing · smelling feet · snatch weight lifting · snorkeling · snowboarding · snowkiting · snowmobiling · somersaulting · spelunking · spinning plates · spinning poi · splashing water · spray painting · spraying · springboard diving · squat · squeezing orange · stacking cups · stacking dice · standing on hands · steer roping · steering car · sticking tongue out ★ · stomping grapes · sucking lolly · surfing water · surveying · sweeping floor · swimming backstroke · swimming breast stroke · swimming butterfly stroke · swimming front crawl · swimming with dolphins · swimming with sharks · swinging baseball bat · swinging on something · sword fighting · sword swallowing · tackling · tagging graffiti · tai chi · taking photo · tapping guitar · tasting beer · tasting food · tasting wine · testifying · threading needle · throwing axe · throwing ball (not baseball or American football) · throwing discus · throwing knife · throwing snowballs · throwing tantrum ★ · throwing water balloon · tickling ★ · tie dying · tightrope walking · tiptoeing · tobogganing · tossing coin · tossing salad · training dog · trapezing · treating wood · trimming or shaving beard · trimming shrubs · trimming trees · triple jump · tying bow tie · tying knot (not on a tie) · tying necktie · tying shoe laces · unboxing · uncorking champagne · unloading truck · using a microscope · using a paint roller · using a power drill · using a sledge hammer · using a wrench · using atm · using bagging machine · using circular saw · using inhaler · using puppets · using remote controller (not gaming) · using segway · vacuuming car · vacuuming floor · visiting the zoo · wading through mud · wading through water · walking on stilts · walking the dog · walking through snow · walking with crutches · washing dishes · washing feet · washing hair · washing hands · water skiing · water sliding · watering plants · waving hand ★ · waxing armpits · waxing back · waxing chest · waxing eyebrows · waxing legs · weaving basket · weaving fabric · welding · whistling · windsurfing · wood burning (art) · wrapping present · wrestling · writing · yarn spinning · yoga · zumba

---

## 5. Retrain runbook \u2014 Family-Concert v1 (server: `~/concert_engagement`)

The new model is trained **on the GPU server** (`sri-gpu-12t4`, `.206`); this Windows box has neither
the features nor the GPUs. It reuses the existing `features_kinetics700` tree **read-only**, so the
original online-learning data and checkpoint are never modified.

**Files (versioned in this repo, copy to the server):**
- `server/labels/hierarchical_labels_family_concert.json` \u2192 place at
  `models/action_transformer_kinetics700/hierarchical_labels_family_concert.json` on the server.
- `scripts/training/patch_family_concert_trainer.py` \u2192 server `scripts/patch_family_concert_trainer.py`.
- `scripts/training/launch_family_concert_v1.sh` \u2192 server `scripts/launch_family_concert_v1.sh`.

**Phase 1 \u2014 train now on the 12 already-extracted classes (no new extraction):**
```bash
cd ~/concert_engagement
bash scripts/launch_family_concert_v1.sh      # detached; prints PID + log path
tail -f logs/train_family_concert_*.log       # sanity-check epoch 1: kept classes + train/val counts
```
The launcher regenerates `scripts/train_family_concert.py` from the patch, filters samples to the
`train_classes` allow-list, up-weights engagement (`--engagement-weight 1.5`), and writes checkpoints
to `models/action_transformer_family_concert_v1/`. `karaoke`/`laughing` are in the allow-list but have
no features yet, so they simply contribute no samples in Phase 1.

**Phase 2 \u2014 add the sing-along/positive cues (optional, bigger):**
1. Extract `karaoke` and `laughing` K700 clips via `scripts/data/extract_kinetics_features.py`.
2. Regenerate `data/processed/{train,val}_samples.npy` + `label_mapping.json` so the new classes get indices.
3. Re-run `scripts/launch_family_concert_v1.sh`.

**Deploy the new checkpoint (drop-in, non-destructive):**
- Copy `models/action_transformer_family_concert_v1/best_model.pth` to the app's model dir. The live app
  (`Handover_JulySession/live_multiperson_binary_v2.py`) resolves `Handover_JulySession/model/best_model.pth`
  first, else `models/action_transformer_12gpus_binary_v2_cleaned/best_model.pth`. Keep the original in place;
  stage the family-concert checkpoint under its own dir.

**Future \u2014 context selection (not yet implemented):** add an app `--context {online-class,family-concert}`
flag that picks the model dir + label file, so the frozen original model serves the online-class context
and this model serves seated family concerts.

---

*Generated 2026-08-18 for the engagement-model retrain discussion. Used-class list and indices are
authoritative from `server/labels/hierarchical_labels.json`; the 700-class catalog is the public
Kinetics-700 label set. Suitability flags are proposals for discussion, not yet applied to any model.*
