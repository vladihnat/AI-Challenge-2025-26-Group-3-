# Metrics

Submissions are ranked primarily by the F1-score: higher F1 is translated into a higher rank.

F1 is the harmonic mean of Precision and Recall. We use it because it provides a single reliability measure that balances:
- Precision (avoid false alarms: predicting “Visitor” when there isn’t one)
- Recall (avoid misses: failing to detect real visitors)

We also report Balanced Accuracy as an additional metric. Balanced Accuracy is the average recall across the two classes (baseline = 0.5). We include it because it prevents inflated performance from models that mostly predict the majority class.

# Phases

The test data is private and split into two evaluation phases:

**Phase 1 — Public (leaderboard)**  
Participants see their scores after each submission based on a test set of 4,000 images. The scores reflects the performance on a public leaderboard.  Throughout the competition, each participant can make up to 100 submissions total. Each submission must not exceed 3600 seconds. 


**Phase 2 — Private (final)**  
Final scores are computed on a hidden portion of the test set (6,000 images) and are revealed only at the end of the competition to prevent leaderboard overfitting/cheating. During this phase, each participant can make up to 10 submissions total. Each submission must not exceed 3600 seconds.
