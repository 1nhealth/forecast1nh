"""Shared test fixtures for pytest."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_referral_data():
    """Create sample referral data for testing."""
    # Create sample data with timestamps
    base_date = datetime(2024, 1, 1, 9, 0, 0)  # Monday 9 AM

    data = {
        'Referral ID': [f'REF{i:03d}' for i in range(1, 21)],
        'TS_Passed_Online_Form': [],
        'TS_Pre-Screening_Activities': [],
        'TS_Sent_To_Site': [],
        'Parsed_Lead_Status_History': []
    }

    # Create timestamps with various patterns
    for i in range(20):
        pof = base_date + timedelta(days=i, hours=i % 12)
        psa = pof + timedelta(hours=2 + (i % 5))  # 2-6 hours later
        sts = psa + timedelta(days=1 + (i % 3))  # 1-3 days later

        data['TS_Passed_Online_Form'].append(pof)
        data['TS_Pre-Screening_Activities'].append(psa)
        data['TS_Sent_To_Site'].append(sts if i % 4 != 0 else pd.NaT)  # Some nulls

        # Create status history
        history = [
            ('Passed Online Form', pof),
            ('Pre-Screening Activities', psa),
        ]
        if pd.notna(sts):
            history.append(('Sent To Site', sts))
        data['Parsed_Lead_Status_History'].append(history)

    return pd.DataFrame(data)


@pytest.fixture
def sample_heatmap_data():
    """Create sample heatmap data (7 days x 24 hours)."""
    # Create a 7x24 matrix with some pattern
    np.random.seed(42)

    contact_heatmap = pd.DataFrame(
        np.random.poisson(10, size=(7, 24)),
        index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        columns=range(24)
    )

    # Higher activity during business hours (9-17)
    for day in range(5):  # Mon-Fri
        for hour in range(9, 17):
            contact_heatmap.iloc[day, hour] *= 2

    # StS heatmap - generally lower numbers, correlated with contacts
    sts_heatmap = pd.DataFrame(
        np.random.poisson(3, size=(7, 24)),
        index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        columns=range(24)
    )

    # Higher success during specific hours
    for day in range(5):
        for hour in range(10, 12):  # Good time: 10-11 AM
            sts_heatmap.iloc[day, hour] *= 3
        for hour in range(16, 18):  # Bad time: 4-5 PM (low success despite contacts)
            sts_heatmap.iloc[day, hour] = max(1, sts_heatmap.iloc[day, hour] // 3)

    return contact_heatmap, sts_heatmap


@pytest.fixture
def business_hours_timestamps():
    """Create timestamps for testing business hours filtering."""
    timestamps = {
        'monday_9am': datetime(2024, 1, 1, 9, 0, 0),      # Monday 9 AM - business hours
        'monday_8am': datetime(2024, 1, 1, 8, 0, 0),      # Monday 8 AM - before hours
        'monday_5pm': datetime(2024, 1, 1, 17, 0, 0),     # Monday 5 PM - end of hours
        'monday_6pm': datetime(2024, 1, 1, 18, 0, 0),     # Monday 6 PM - after hours
        'friday_2pm': datetime(2024, 1, 5, 14, 0, 0),     # Friday 2 PM - business hours
        'saturday_10am': datetime(2024, 1, 6, 10, 0, 0),  # Saturday 10 AM - weekend
        'sunday_10am': datetime(2024, 1, 7, 10, 0, 0),    # Sunday 10 AM - weekend
    }
    return timestamps
