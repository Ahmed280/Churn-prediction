"""
Synthetic-data test for the event processor.

Generates a small but realistic event log to assert that the processor yields
both classes (active/churn) with a sane prevalence range (5–50%).

Author: Ahmed Alghaith
Date: August 2025.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import our fixed processor
from MusicStreamingEventProcessor import MusicStreamingEventProcessor

def create_test_data():
    """Create reproducible synthetic event logs for unit tests."""
    print("📊 Creating test data...")

    np.random.seed(42)
    sample_events = []
    users = [f"user_{i}" for i in range(50)]  # Smaller dataset for testing

    # Define realistic page types
    pages = ['NextSong', 'Home', 'Thumbs Up', 'Thumbs Down', 'Settings',
             'Logout', 'Help', 'About', 'Add Friend', 'Add to Playlist']

    base_time = int(datetime(2024, 1, 1).timestamp() * 1000)
    end_time = int(datetime(2024, 4, 1).timestamp() * 1000)  # 3 months of data

    # Create events with realistic patterns for churn
    for user_idx, user_id in enumerate(users):
        # Vary activity levels to create natural churn patterns
        user_activity_level = np.random.choice(
            ['high', 'medium', 'low', 'churning'],
            p=[0.3, 0.4, 0.2, 0.1]
        )

        if user_activity_level == 'high':
            n_events = np.random.randint(100, 300)
        elif user_activity_level == 'medium':
            n_events = np.random.randint(30, 100)
        elif user_activity_level == 'low':
            n_events = np.random.randint(10, 30)
        else:  # churning
            n_events = np.random.randint(5, 15)

        # Registration time
        reg_time = base_time + np.random.randint(0, 30 * 24 * 3600 * 1000)

        # Generate events
        for i in range(n_events):
            # Churning users have events only in first part of timeline
            if user_activity_level == 'churning':
                event_time = reg_time + np.random.randint(0, 20 * 24 * 3600 * 1000)  # First 20 days only
            else:
                event_time = reg_time + np.random.randint(0, end_time - reg_time)

            event = {
                'ts': event_time,
                'userId': user_id,
                'page': np.random.choice(pages, p=[0.5, 0.15, 0.1, 0.05, 0.05, 0.04, 0.03, 0.02, 0.03, 0.03]),
                'sessionId': user_idx * 5 + np.random.randint(1, 5),
                'level': np.random.choice(['free', 'paid'], p=[0.7, 0.3]),
                'gender': np.random.choice(['M', 'F'])
            }
            sample_events.append(event)

    events_df = pd.DataFrame(sample_events)
    events_df = events_df.sort_values('ts').reset_index(drop=True)

    print(f"   ✅ Created {len(events_df):,} events for {events_df['userId'].nunique()} users")
    return events_df

def test_processor():
    """End-to-end assertion: clean → engineer features → label → sanity checks."""
    print("🧪 Testing Fixed MusicStreamingEventProcessor...")

    # Create test data
    events_df = create_test_data()

    # Initialize processor
    processor = MusicStreamingEventProcessor(
        prediction_horizon_days=7,
        inactive_threshold_days=20  # Shorter for test data
    )

    # Step 1: Clean events
    cleaned_events = processor.clean_events(events_df)

    # Step 2: Engineer features
    user_features = processor.engineer_user_features()

    # Step 3: Identify churn
    churn_labels = processor.identify_churn_users()

    # Step 4: Attach churn labels
    user_features['churn'] = user_features['userId'].map(churn_labels)
    user_features['churn'] = user_features['churn'].fillna(0).astype(int)

    # Analyze results
    print(f"\n📊 TEST RESULTS:")
    print(f"   Total users: {len(user_features)}")
    print(f"   Features created: {len([col for col in user_features.columns if col not in ['userId', 'churn']])}")

    # Check class distribution
    churn_dist = user_features['churn'].value_counts().sort_index()
    print(f"\n🎯 Class Distribution:")
    for class_label, count in churn_dist.items():
        class_name = 'Active' if class_label == 0 else 'Churned'
        percentage = count / len(user_features) * 100
        print(f"   {class_name} (class {class_label}): {count} users ({percentage:.1f}%)")

    # Success criteria
    unique_classes = user_features['churn'].nunique()
    has_both_classes = 0 in churn_dist.index and 1 in churn_dist.index
    churn_rate = user_features['churn'].mean()

    print(f"\n✅ TEST VALIDATION:")
    print(f"   Unique classes: {unique_classes} {'✅' if unique_classes >= 2 else '❌'}")
    print(f"   Has both classes: {has_both_classes} {'✅' if has_both_classes else '❌'}")
    print(f"   Churn rate: {churn_rate:.1%} {'✅' if 0.05 <= churn_rate <= 0.5 else '❌'}")

    if unique_classes >= 2 and has_both_classes and 0.05 <= churn_rate <= 0.5:
        print(f"\n🎉 SUCCESS! Processor creates balanced classes.")
        print(f"   Ready for model training with both classes.")
        return True
    else:
        print(f"\n❌ FAILED! Processor still has class issues.")
        return False

if __name__ == "__main__":
    success = test_processor()
    if success:
        print("\n✅ Fixed processor is working correctly!")
    else:
        print("\n❌ Processor still needs fixes.")
