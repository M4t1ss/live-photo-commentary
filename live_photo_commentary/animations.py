import random
import re
import base64
from collections import Counter
from itertools import combinations
from functools import lru_cache


import yaml
from PIL import Image


NEG_INF = float('-inf')
POS_INF = float('inf')

class Animation:
    def __init__(self, filename, setname=None):
        self.filename = filename
        self.setname = setname

        img = Image.open(filename)
        self.width, self.height = img.size
        self.duration = 0
        while True:
            self.duration += img.info.get('duration', 0)  # duration in ms
            try:
                img.seek(img.tell() + 1)
            except EOFError:
                break

    def __lt__(self, other):
        if not isinstance(other, Animation):
            return NotImplemented
        return self.filename < other.filename

    @staticmethod
    @lru_cache(maxsize=None)
    def load(filename, setname):
        return Animation(filename, setname)

    def data_uri(self):
        """Convert animation file to data URI."""
        with open(self.filename, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            ext = self.filename.split(".")[-1]
            return f"data:image/{ext};base64,{b64}"

    def __repr__(self):
        return f'<Animation "{self.filename}" ({self.duration / 1000:.2f}s)>'


class AnimatedThing:
    def __init__(self, animations):
        self.animations = animations

    def pick(self):
        return random.choice(self.animations)

class AnimatedTrigger(AnimatedThing):
    def __init__(self, animations, regex):
        super().__init__(animations)
        self.regex = re.compile(regex, re.I)

class AnimatedButton(AnimatedThing):
    def __init__(self, animations, label):
        super().__init__(animations)
        self.label = label


class Animations:
    def __init__(self, source, safe_distance=30, prune_size=10):
        self.safe_distance = safe_distance
        self.prune_size = prune_size

        if isinstance(source, dict):
            data = source
        elif isinstance(source, str):
            with open(source, 'rt') as r:
                data = yaml.safe_load(r)
        else:
            data = yaml.safe_load(source)

        self.animation_sets = {
            setname: [
                Animation.load(filename, setname)
                for filename in filenames
            ]
            for setname, filenames in data['animations'].items()
        }
        self.triggers = self._make_animated(AnimatedTrigger, data['triggers'])
        self.buttons = self._make_animated(AnimatedButton, data['buttons'])
        self.system = dict(zip(
            data['system'].keys(),
            self._make_animated(AnimatedThing, data['system'].values()),
        ))

        self.max_height = max(
            animation.height
            for animations in self.animation_sets.values()
            for animation in animations
        )

    def _make_animated(self, klass, data):
        result = []
        for item in data:
            animationset_names = item.pop('animations')
            animations = [
                animation
                for name in animationset_names
                for animation in self.animation_sets[name]
            ]
            result.append(klass(animations, **item))
        return result

    def _score(self, repeat_score, streak_score, set_count_score, set_streak_score, proximity_score):
        score = sum([
            # int from 1
            repeat_score * 1.0,

            # int from 0
            streak_score * 2.0,

            # int from 1
            set_count_score * 0.5,

            # int from 0
            set_streak_score * 1.0,

            # float from 0 to 1
            proximity_score * 5.0,
        ])
        return score

    def _find_all_matches(self, text):
        """Find all animation matches in text, sorted by position."""
        return sorted([
            (match.start(), trigger.pick())
            for trigger in self.triggers
            for match in trigger.regex.finditer(text)
        ], key=lambda x: x[0]) # Add the key here

    def _pad_with_talking(self, found_animations, text):
        """Add talking animations to reach prune_size if needed."""
        if len(found_animations) >= self.prune_size:
            return found_animations

        num_to_add = self.prune_size - len(found_animations)

        text_length = len(text)
        if num_to_add == 1:
            positions = [text_length // 2]
        else:
            positions = [int(i * text_length / (num_to_add - 1)) for i in range(num_to_add)]

        talking_additions = [
            (pos, random.choice(self.animation_sets['talking']))
            for pos in positions
        ]

        return sorted(found_animations + talking_additions, key=lambda x: x[0]) # Add the key here

    def _calculate_total_duration(self, combo):
        """Calculate total duration of animations in milliseconds."""
        return sum(animation.duration for _, animation in combo)

    def _is_tight_fit(self, found_animations, total_duration, duration):
        """Check if animations are a tight fit: can't remove any without going under duration."""
        leeway = total_duration - duration
        return all(
            animation.duration >= leeway
            for _, animation in found_animations
        )

    def _needs_pruning(self, found_animations, duration):
        """Check if we need to prune animations."""
        total_duration = self._calculate_total_duration(found_animations)
        if total_duration < duration:
            # The animations are too short anyway
            return False

        # Don't prune if already a tight fit
        return not self._is_tight_fit(found_animations, total_duration, duration)

    def _score_animations(self, found_animations, include_streaks=False):
        """Score animations. Returns list of individual scores."""
        if include_streaks:
            file_streaks, set_streaks = self._calculate_streaks(found_animations)
        else:
            file_streaks = [0] * len(found_animations)
            set_streaks = [0] * len(found_animations)

        file_counts = Counter(animation for _, animation in found_animations)
        set_counts = Counter(animation.setname for _, animation in found_animations)
        scores = []
        prev_start = 0

        for i, (start, animation) in enumerate(found_animations):
            repeat_score = file_counts[animation]
            streak_score = file_streaks[i]
            set_count_score = set_counts[animation.setname]
            set_streak_score = set_streaks[i]
            proximity_score = 1.0 if i == 0 else 1.0 - min(start - prev_start, self.safe_distance) / self.safe_distance
            score = self._score(repeat_score, streak_score, set_count_score, set_streak_score, proximity_score)
            scores.append(score)
            prev_start = start

        return scores

    def _score_for_pruning(self, found_animations):
        """Score animations for initial pruning based on repetition and proximity."""
        scores = self._score_animations(found_animations, include_streaks=False)
        return [(score, ix) for ix, score in enumerate(scores)]

    def _prune_to_top(self, found_animations, scores, n=None):
        """Prune to top N animations by score."""
        if n is None:
            n = self.prune_size
        top_ixs = sorted(ix for _, ix in sorted(scores)[:n])
        return [found_animations[ix] for ix in top_ixs]

    def _calculate_streaks(self, combo):
        """Calculate streak scores for a combination. Returns (file_streaks, set_streaks)."""
        file_streaks = [0] * len(combo)
        set_streaks = [0] * len(combo)
        for i in range(1, len(combo)):
            if combo[i][1] == combo[i - 1][1]:
                file_streaks[i] = file_streaks[i - 1] + 1
            if combo[i][1].setname == combo[i - 1][1].setname:
                set_streaks[i] = set_streaks[i - 1] + 1
        return file_streaks, set_streaks

    def _score_combination(self, combo):
        """Calculate full score for a combination including streaks."""
        return sum(self._score_animations(combo, include_streaks=True))

    def _is_valid_combination(self, combo, duration):
        """Check if a combination is valid (fits duration and can't be reduced further)."""
        total_duration = self._calculate_total_duration(combo)
        if total_duration < duration:
            return False
        return self._is_tight_fit(combo, total_duration, duration)

    def _find_best_combination(self, pruned, duration):
        """Find the best combination from pruned animations by lowest score."""
        best_combo = []
        best_score = POS_INF

        for r in range(1, len(pruned) + 1):
            for combo in combinations(pruned, r):
                if not self._is_valid_combination(combo, duration):
                    continue

                total_score = self._score_combination(combo)

                if total_score < best_score:
                    best_combo = combo
                    best_score = total_score

        return best_combo

    def find_animations(self, text, duration):
        duration = duration * 1000  # convert to milliseconds

        # Find all matches
        found_animations = self._find_all_matches(text)

        # Pad with talking animations if needed
        found_animations = self._pad_with_talking(found_animations, text)

        # Check if we need to reduce the list
        if not self._needs_pruning(found_animations, duration):
            return [animation for _, animation in found_animations]

        # Score and prune to top animations
        scores = self._score_for_pruning(found_animations)
        pruned = self._prune_to_top(found_animations, scores)

        # Find best combination with full scoring
        best_combo = self._find_best_combination(pruned, duration)

        return [animation for _, animation in best_combo]

    def pick(self, setname):
        return self.system[setname].pick()


                


if __name__ == '__main__':
    random.seed(42)
    animations = Animations('animations.yaml')
    print(animations.pick('start'))
    text = 'Hello! I am very scared! Would you like to wave to me? Very cute! Are you happy with this?'
    text_animations = animations.find_animations(text, 25)
    print(text_animations)
