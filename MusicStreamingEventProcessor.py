"""
Working Music Streaming Event Processor (Based on Proven Notebook)

This is the PROVEN implementation from Full_process-1.ipynb that successfully
creates balanced classes and works with all models.

Author: Ahmed Alghaith
Date: August 2025
"""

from utils import *
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union

class MusicStreamingEventProcessor:
    """Leak-safe processor for event-based churn problem.

    Args:
        prediction_horizon_days: Future window used to define the cutoff date for labeling.
        inactive_threshold_days: Number of days without activity to mark a user as churned.

    Attributes:
        events_df: Raw events (after loading).
        cleaned_events_df: Events after cleaning & leakage removal.
        user_features_df: User-level feature table.
        churn_definition_method: String identifier of the labeling method.
    """
    def __init__(self, prediction_horizon_days: int = 7, inactive_threshold_days: int = 30):
        self.events_df = None
        self.cleaned_events_df = None
        self.user_features_df = None
        self.churn_definition_method = None
        self.prediction_horizon_days = prediction_horizon_days
        self.inactive_threshold_days = inactive_threshold_days

    def clean_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw event logs and remove leakage sources.

        - Converts timestamps to datetime.
        - Drops rows with missing `userId`.
        - **Removes explicit churn page events** (prevents label leakage).
        - Normalizes `userId` type.

        Args:
            events_df: Raw event log with at least `ts`, `userId`, `page`.

        Returns:
            Cleaned DataFrame with datetime column and safe pages only.
        """
        print("🧹 Cleaning event data...")
        self.events_df = events_df.copy()

        # Convert timestamp to datetime
        self.events_df['datetime'] = pd.to_datetime(self.events_df['ts'], unit='ms')

        # Remove rows with missing userId
        initial_count = len(self.events_df)
        self.events_df = self.events_df.dropna(subset=['userId'])
        removed_missing = initial_count - len(self.events_df)

        # CRITICAL: Remove explicit churn events to prevent data leakage
        churn_events = [
            'Cancellation Confirmation', 'Downgrade', 'Submit Downgrade',
            'Cancel', 'Unsubscribe', 'Submit Cancel'
        ]
        initial_count = len(self.events_df)
        self.events_df = self.events_df[~self.events_df['page'].isin(churn_events)]
        removed_churn = initial_count - len(self.events_df)

        # Convert userId to string for consistency
        self.events_df['userId'] = self.events_df['userId'].astype(str)

        self.cleaned_events_df = self.events_df.copy()

        print(f"   ✅ Cleaning results:")
        print(f"      Removed {removed_missing} events with missing userId")
        if removed_churn > 0:
            print(f"      🚫 Removed {removed_churn} explicit churn events (data leakage prevention)")
        print(f"      Final events: {len(self.cleaned_events_df):,}")
        print(f"      Unique users: {self.cleaned_events_df['userId'].nunique()}")
        print(f"      Date range: {self.cleaned_events_df['datetime'].min()} to {self.cleaned_events_df['datetime'].max()}")

        return self.cleaned_events_df

    def identify_churn_users(self, cutoff_date: Optional[datetime] = None) -> Dict[str, int]:
        """Create labels from inactivity (no lookahead leakage).

        A user is churned if last activity < (cutoff_date - inactive_threshold_days).

        Args:
            cutoff_date: Optional cutoff (defaults to max(datetime) - prediction_horizon_days).

        Returns:
            Mapping of `userId` → 0/1 label for users seen up to the cutoff.
        """
        if self.cleaned_events_df is None:
            raise ValueError("Must clean events data first using clean_events()")

        print(f"🎯 Identifying churned users using PROVEN activity-based method")
        print(f"   Prediction horizon: {self.prediction_horizon_days} days")
        print(f"   Inactivity threshold: {self.inactive_threshold_days} days")

        # If no cutoff date provided, use the end of data minus prediction horizon
        if cutoff_date is None:
            max_date = self.cleaned_events_df['datetime'].max()
            cutoff_date = max_date - timedelta(days=self.prediction_horizon_days)
        print(f"   Cutoff date for prediction: {cutoff_date}")

        # Only use events up to the cutoff date for feature calculation
        events_for_features = self.cleaned_events_df[
            self.cleaned_events_df['datetime'] <= cutoff_date
        ]
        if len(events_for_features) == 0:
            print("   ⚠️ No events found before cutoff date")
            return {}

        # Find last activity date for each user
        user_last_activity = events_for_features.groupby('userId')['datetime'].max()

        # Users are churned if their last activity is before the threshold
        churn_threshold = cutoff_date - timedelta(days=self.inactive_threshold_days)
        churned_users = user_last_activity[user_last_activity < churn_threshold].index

        # Create churn labels for all users present up to cutoff
        all_users = events_for_features['userId'].unique()
        churn_labels = {}
        for user in all_users:
            churn_labels[user] = 1 if user in churned_users else 0

        churn_count = sum(churn_labels.values())
        total_users = len(churn_labels)
        print(f"   📊 Churn analysis results:")
        print(f"      Total users: {total_users}")
        print(f"      Churned users: {churn_count} ({churn_count/total_users:.2%})")
        print(f"      Active users: {total_users - churn_count} ({1 - churn_count/total_users:.2%})")

        self.churn_definition_method = 'activity_based_no_leakage'
        return churn_labels

    def engineer_user_features(self, cutoff_date: Optional[datetime] = None) -> pd.DataFrame:
        """Aggregate per-user features using only events up to cutoff.

        Features include: activity counts, engagement ratios, subscription mix,
        weekday/weekend patterns, peak hour, and session variety.

        Args:
            cutoff_date: Optional cutoff to avoid leakage in feature computation.

        Returns:
            User-level feature table with `userId` and numeric columns.
        """
        if self.cleaned_events_df is None:
            raise ValueError("Must clean events data first using clean_events()")

        print("🔧 Engineering comprehensive user features...")

        # Use only events up to cutoff date to prevent leakage
        if cutoff_date is not None:
            events_df = self.cleaned_events_df[
                self.cleaned_events_df['datetime'] <= cutoff_date
            ].copy()
            print(f"   Using events up to {cutoff_date} (n={len(events_df):,})")
        else:
            events_df = self.cleaned_events_df.copy()

        # Get unique users and initialize list for feature dicts
        users = events_df['userId'].unique()
        user_features = []
        print(f"   Processing {len(users)} users...")

        for user_id in users:
            user_events = events_df[events_df['userId'] == user_id].copy()
            features = self._calculate_user_features(user_events, user_id)
            user_features.append(features)

        user_df = pd.DataFrame(user_features)
        self.user_features_df = user_df
        print(f"   ✅ Engineered {len(user_df.columns)} features for {len(user_df)} users")

        return user_df

    def _calculate_user_features(self, user_events: pd.DataFrame, user_id: str) -> Dict[str, Any]:
        """Calculate features using PROVEN method from working notebook."""
        features = {'userId': user_id}
        if len(user_events) == 0:
            # Return default values for users with no events
            return {**features, **{
                'total_events': 0, 'unique_sessions': 0, 'total_songs_played': 0,
                'avg_session_length': 0, 'days_active': 0, 'thumbs_up': 0,
                'thumbs_down': 0, 'home_visits': 0, 'settings_visits': 0,
                'help_visits': 0, 'add_friend': 0, 'add_playlist': 0,
                'engagement_ratio': 0, 'avg_daily_events': 0,
                'paid_events_ratio': 0, 'last_level_paid': 0,
                'weekend_activity_ratio': 0, 'peak_hour': 12, 'session_variety': 0
            }}

        # Basic activity features
        song_events = user_events[user_events['page'] == 'NextSong']
        features.update({
            'total_events': len(user_events),
            'unique_sessions': user_events['sessionId'].nunique() if 'sessionId' in user_events.columns else 1,
            'total_songs_played': len(song_events),
            'avg_session_length': len(user_events) / max(1, user_events['sessionId'].nunique()) if 'sessionId' in user_events.columns else len(user_events),
            'days_active': max(1, (user_events['datetime'].max() - user_events['datetime'].min()).days + 1),
        })

        # Engagement features
        features.update({
            'thumbs_up': len(user_events[user_events['page'] == 'Thumbs Up']),
            'thumbs_down': len(user_events[user_events['page'] == 'Thumbs Down']),
            'home_visits': len(user_events[user_events['page'] == 'Home']),
            'settings_visits': len(user_events[user_events['page'] == 'Settings']),
            'help_visits': len(user_events[user_events['page'] == 'Help']),
            'add_friend': len(user_events[user_events['page'] == 'Add Friend']),
            'add_playlist': len(user_events[user_events['page'] == 'Add to Playlist']),
        })

        # Derived engagement metrics
        total_engagement = (features['thumbs_up'] + features['thumbs_down']
                            + features.get('add_friend', 0) + features.get('add_playlist', 0))
        features['engagement_ratio'] = total_engagement / max(1, features['total_events'])
        features['avg_daily_events'] = features['total_events'] / max(1, features['days_active'])

        # Subscription features (safe - no explicit churn events included)
        if 'level' in user_events.columns:
            paid_events = user_events[user_events['level'] == 'paid']
            features.update({
                'paid_events_ratio': len(paid_events) / len(user_events) if len(user_events) > 0 else 0,
                'last_level_paid': 1 if len(paid_events) > 0 else 0
            })
        else:
            features.update({'paid_events_ratio': 0, 'last_level_paid': 0})

        # Temporal patterns
        user_events_temp = user_events.copy()
        user_events_temp['hour'] = user_events_temp['datetime'].dt.hour
        user_events_temp['day_of_week'] = user_events_temp['datetime'].dt.dayofweek
        user_events_temp['is_weekend'] = user_events_temp['day_of_week'].isin([5, 6])
        features.update({
            'weekend_activity_ratio': user_events_temp['is_weekend'].mean(),
            'peak_hour': user_events_temp['hour'].mode().iloc[0] if len(user_events_temp) > 0 else 12,
            'session_variety': user_events_temp['page'].nunique(),
        })

        return features
