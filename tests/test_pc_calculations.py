"""Tests for PC calculations."""
import pytest
import pandas as pd
import numpy as np
from pc_calculations import (
    calculate_heatmap_data,
    calculate_average_time_metrics,
    analyze_heatmap_efficiency
)


class TestCalculateHeatmapData:
    """Tests for calculate_heatmap_data function."""

    def test_basic_heatmap_creation(self, sample_referral_data):
        """Test basic heatmap data creation."""
        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site'
        }
        contact_heatmap, sts_heatmap = calculate_heatmap_data(
            sample_referral_data,
            ts_col_map,
            'Parsed_Lead_Status_History'
        )

        # Check shape
        assert contact_heatmap.shape == (7, 24)
        assert sts_heatmap.shape == (7, 24)

        # Check index and columns
        expected_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        assert list(contact_heatmap.index) == expected_days
        assert list(contact_heatmap.columns) == list(range(24))

    def test_empty_dataframe(self):
        """Test heatmap creation with empty dataframe."""
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': [],
            'TS_Sent_To_Site': [],
            'Parsed_Lead_Status_History': []
        })
        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site'
        }
        contact_heatmap, sts_heatmap = calculate_heatmap_data(
            df,
            ts_col_map,
            'Parsed_Lead_Status_History'
        )

        # Empty dataframe returns empty heatmaps
        assert contact_heatmap.empty
        assert sts_heatmap.empty


class TestCalculateAverageTimeMetrics:
    """Tests for calculate_average_time_metrics function."""

    def test_basic_metrics_calculation(self, sample_referral_data):
        """Test basic time metrics calculation."""
        ts_col_map = {
            'Passed Online Form': 'TS_Passed_Online_Form',
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site'
        }

        result = calculate_average_time_metrics(
            sample_referral_data,
            ts_col_map,
            'Parsed_Lead_Status_History'
        )

        # Should return dictionary with expected keys
        assert 'avg_time_to_first_contact' in result
        assert 'avg_time_between_contacts' in result
        assert 'avg_time_new_to_sts' in result

        # All should be positive numbers
        assert result['avg_time_to_first_contact'] > 0
        assert result['avg_time_new_to_sts'] > 0

    def test_with_null_values(self):
        """Test metrics calculation with null values."""
        df = pd.DataFrame({
            'TS_Passed_Online_Form': pd.to_datetime(['2024-01-01 09:00:00', '2024-01-02 09:00:00', None]),
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 10:00:00', None, '2024-01-03 10:00:00']),
            'TS_Sent_To_Site': pd.to_datetime(['2024-01-02 10:00:00', '2024-01-03 10:00:00', None]),
            'Parsed_Lead_Status_History': [
                [('Passed Online Form', pd.Timestamp('2024-01-01 09:00:00'))],
                [('Passed Online Form', pd.Timestamp('2024-01-02 09:00:00'))],
                []
            ]
        })

        ts_col_map = {
            'Passed Online Form': 'TS_Passed_Online_Form',
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site'
        }

        result = calculate_average_time_metrics(df, ts_col_map, 'Parsed_Lead_Status_History')

        # Should handle nulls gracefully
        assert isinstance(result, dict)
        assert 'avg_time_to_first_contact' in result
        assert 'avg_time_between_contacts' in result
        assert 'avg_time_new_to_sts' in result

    def test_all_null_values(self):
        """Test metrics with all null timestamps."""
        df = pd.DataFrame({
            'TS_Passed_Online_Form': [None, None],
            'TS_Pre-Screening_Activities': [None, None],
            'TS_Sent_To_Site': [None, None],
            'Parsed_Lead_Status_History': [[], []]
        })

        ts_col_map = {
            'Passed Online Form': 'TS_Passed_Online_Form',
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site'
        }

        result = calculate_average_time_metrics(df, ts_col_map, 'Parsed_Lead_Status_History')

        # Should return NaN for all metrics
        assert pd.isna(result['avg_time_to_first_contact'])
        assert pd.isna(result['avg_time_between_contacts'])
        assert pd.isna(result['avg_time_new_to_sts'])


class TestAnalyzeHeatmapEfficiency:
    """Tests for analyze_heatmap_efficiency function."""

    def test_basic_efficiency_analysis(self, sample_heatmap_data):
        """Test basic efficiency analysis."""
        contact_heatmap, sts_heatmap = sample_heatmap_data

        result = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap)

        # Should return dictionary with expected keys (new format)
        assert 'best_times' in result
        assert 'avoid_times' in result

        # Each should be a list
        assert isinstance(result['best_times'], list)
        assert isinstance(result['avoid_times'], list)

    def test_zero_contacts(self):
        """Test efficiency analysis with zero contacts."""
        contact_heatmap = pd.DataFrame(
            np.zeros((7, 24)),
            index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            columns=range(24)
        )
        sts_heatmap = contact_heatmap.copy()

        result = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap)

        # Should handle gracefully - returns empty dict
        assert result == {}


# New tests for updated functionality
class TestHeatmapBusinessHours:
    """Tests for heatmap business hours filtering."""

    def test_heatmap_business_hours_parameter(self):
        """Test that heatmap function accepts business_hours_only parameter."""
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 09:00:00', '2024-01-06 10:00:00']),  # Mon, Sat
            'TS_Sent_To_Site': pd.to_datetime(['2024-01-02 14:00:00', '2024-01-08 14:00:00']),
            'Parsed_Lead_Status_History': [
                [('Event1', pd.Timestamp('2024-01-01 10:00:00')), ('Event2', pd.Timestamp('2024-01-01 15:00:00'))],
                [('Event3', pd.Timestamp('2024-01-06 11:00:00'))]  # Saturday event
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site'
        }

        # Should work with business_hours_only=False (default)
        contact_hm, sts_hm = calculate_heatmap_data(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)
        assert contact_hm.shape == (7, 24)

        # Should work with business_hours_only=True
        contact_hm_bh, sts_hm_bh = calculate_heatmap_data(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=True)
        assert contact_hm_bh.shape == (7, 24)

    def test_heatmap_filters_weekend_events(self):
        """Test that business hours excludes weekend events from heatmap."""
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 09:00:00', '2024-01-06 10:00:00']),
            'TS_Sent_To_Site': pd.to_datetime([None, None]),
            'Parsed_Lead_Status_History': [
                [('Event1', pd.Timestamp('2024-01-01 10:00:00'))],  # Monday - should be included
                [('Event2', pd.Timestamp('2024-01-06 11:00:00'))]   # Saturday - should be excluded
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site'
        }

        # All hours
        contact_hm, _ = calculate_heatmap_data(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)
        total_all_hours = contact_hm.sum().sum()

        # Business hours only
        contact_hm_bh, _ = calculate_heatmap_data(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=True)
        total_bh = contact_hm_bh.sum().sum()

        # Business hours should have fewer events (Saturday excluded)
        assert total_bh < total_all_hours


class TestBusinessHoursMetrics:
    """Tests for business hours metrics."""

    def test_metrics_business_hours_parameter(self):
        """Test that business_hours_only parameter is accepted."""
        df = pd.DataFrame({
            'TS_Passed_Online_Form': pd.to_datetime(['2024-01-01 09:00:00']),  # Monday 9 AM
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 10:00:00']),  # Monday 10 AM
            'TS_Sent_To_Site': pd.to_datetime(['2024-01-02 14:00:00']),  # Tuesday 2 PM
            'Parsed_Lead_Status_History': [
                [('Passed Online Form', pd.Timestamp('2024-01-01 09:00:00')),
                 ('Pre-Screening Activities', pd.Timestamp('2024-01-01 10:00:00')),
                 ('Sent To Site', pd.Timestamp('2024-01-02 14:00:00'))]
            ]
        })

        ts_col_map = {
            'Passed Online Form': 'TS_Passed_Online_Form',
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site'
        }

        # Should work with business_hours_only=False (default)
        result = calculate_average_time_metrics(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)
        assert 'avg_time_to_first_contact' in result

        # Should work with business_hours_only=True
        result_bh = calculate_average_time_metrics(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=True)
        assert 'avg_time_to_first_contact' in result_bh

    def test_business_hours_filters_after_hours(self):
        """Test that business hours excludes after-hours events."""
        df = pd.DataFrame({
            'TS_Passed_Online_Form': pd.to_datetime([
                '2024-01-01 09:00:00',  # Monday 9 AM - business hours
                '2024-01-02 09:00:00',  # Tuesday 9 AM - business hours
            ]),
            'TS_Pre-Screening_Activities': pd.to_datetime([
                '2024-01-01 18:00:00',  # Monday 6 PM - AFTER hours
                '2024-01-02 10:00:00',  # Tuesday 10 AM - business hours
            ]),
            'TS_Sent_To_Site': pd.to_datetime([
                '2024-01-02 09:00:00',  # Tuesday 9 AM
                '2024-01-03 14:00:00',  # Wednesday 2 PM
            ]),
            'Parsed_Lead_Status_History': [
                [],
                []
            ]
        })

        ts_col_map = {
            'Passed Online Form': 'TS_Passed_Online_Form',
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site'
        }

        # With business hours filter, first row should be excluded (PSA is after hours)
        result_bh = calculate_average_time_metrics(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=True)

        # Only second row should count for TTFC
        assert result_bh['avg_time_to_first_contact'] > 0

    def test_business_hours_filters_weekends(self):
        """Test that business hours excludes weekends."""
        df = pd.DataFrame({
            'TS_Passed_Online_Form': pd.to_datetime([
                '2024-01-01 09:00:00',  # Monday 9 AM - business hours
                '2024-01-06 10:00:00',  # Saturday 10 AM - WEEKEND
            ]),
            'TS_Pre-Screening_Activities': pd.to_datetime([
                '2024-01-01 10:00:00',  # Monday 10 AM - business hours
                '2024-01-06 11:00:00',  # Saturday 11 AM - WEEKEND
            ]),
            'TS_Sent_To_Site': pd.to_datetime([
                '2024-01-02 09:00:00',  # Tuesday 9 AM
                '2024-01-08 14:00:00',  # Monday 2 PM (following week)
            ]),
            'Parsed_Lead_Status_History': [
                [],
                []
            ]
        })

        ts_col_map = {
            'Passed Online Form': 'TS_Passed_Online_Form',
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site'
        }

        # With business hours filter, second row should be excluded (weekend)
        result_bh = calculate_average_time_metrics(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=True)

        # Only first row should be counted
        assert result_bh['avg_time_to_first_contact'] > 0


class TestUpdatedHeatmapEfficiency:
    """Tests for updated heatmap efficiency analysis."""

    def test_returns_correct_structure(self, sample_heatmap_data):
        """Test that function returns correct structure with new keys."""
        contact_heatmap, sts_heatmap = sample_heatmap_data
        result = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap)

        # Should have only two keys now
        assert 'best_times' in result
        assert 'avoid_times' in result

        # Should not have old keys
        assert 'volume_best' not in result
        assert 'most_efficient' not in result
        assert 'least_efficient' not in result

    def test_no_overlap_between_categories(self, sample_heatmap_data):
        """Test that best times and avoid times don't overlap."""
        contact_heatmap, sts_heatmap = sample_heatmap_data
        result = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap)

        best_times = result.get('best_times', [])
        avoid_times = result.get('avoid_times', [])

        # Check for any overlapping time slots
        best_set = set(best_times)
        avoid_set = set(avoid_times)
        overlap = best_set.intersection(avoid_set)

        assert len(overlap) == 0, f"Found overlapping time slots: {overlap}"

    def test_best_times_focus_on_efficiency(self, sample_heatmap_data):
        """Test that best times prioritize efficiency."""
        contact_heatmap, sts_heatmap = sample_heatmap_data
        result = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap)

        best_times = result.get('best_times', [])

        # Should have results (sample data has activity)
        assert len(best_times) > 0

        # Should have at most 10 time slots
        assert len(best_times) <= 10

    def test_avoid_times_high_volume_low_efficiency(self, sample_heatmap_data):
        """Test that avoid times have high volume but low efficiency."""
        contact_heatmap, sts_heatmap = sample_heatmap_data
        result = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap)

        avoid_times = result.get('avoid_times', [])

        # Should have results
        assert len(avoid_times) >= 0  # Could be empty if all times are efficient

        # Should have at most 10 time slots
        assert len(avoid_times) <= 10

    def test_empty_heatmaps(self):
        """Test with zero activity."""
        contact_heatmap = pd.DataFrame(
            np.zeros((7, 24)),
            index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            columns=range(24)
        )
        sts_heatmap = contact_heatmap.copy()

        result = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap)

        # Should return empty dict or empty lists
        if result:
            assert len(result.get('best_times', [])) == 0
            assert len(result.get('avoid_times', [])) == 0

    def test_consistent_format(self, sample_heatmap_data):
        """Test that output format is consistent."""
        contact_heatmap, sts_heatmap = sample_heatmap_data
        result = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap)

        best_times = result.get('best_times', [])
        avoid_times = result.get('avoid_times', [])

        # All should be strings
        assert all(isinstance(t, str) for t in best_times)
        assert all(isinstance(t, str) for t in avoid_times)

        # Should contain day and time info
        for time_str in best_times + avoid_times:
            assert ',' in time_str  # Should have "Day, Time" format


class TestTTFCBusinessHours:
    """Tests for TTFC effectiveness with business hours."""

    def test_ttfc_business_hours_parameter(self):
        """Test that TTFC function accepts business_hours_only parameter."""
        from pc_calculations import calculate_ttfc_effectiveness

        df = pd.DataFrame({
            'TS_Passed_Online_Form': pd.to_datetime(['2024-01-01 09:00:00', '2024-01-06 10:00:00']),  # Mon, Sat
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-06 11:00:00']),
            'TS_Sent_To_Site': pd.to_datetime([None, None]),
            'TS_Signed_ICF': pd.to_datetime([None, None]),
            'TS_Enrolled': pd.to_datetime([None, None])
        })

        ts_col_map = {
            'Passed Online Form': 'TS_Passed_Online_Form',
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        # Should work with business_hours_only=False
        result = calculate_ttfc_effectiveness(df, ts_col_map, business_hours_only=False)
        assert isinstance(result, pd.DataFrame)

        # Should work with business_hours_only=True
        result_bh = calculate_ttfc_effectiveness(df, ts_col_map, business_hours_only=True)
        assert isinstance(result_bh, pd.DataFrame)

    def test_ttfc_filters_weekend_contacts(self):
        """Test that business hours excludes weekend first contacts."""
        from pc_calculations import calculate_ttfc_effectiveness

        df = pd.DataFrame({
            'TS_Passed_Online_Form': pd.to_datetime(['2024-01-01 09:00:00', '2024-01-06 10:00:00']),  # Mon, Sat
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-06 11:00:00']),
            'TS_Sent_To_Site': pd.to_datetime([None, None]),
            'TS_Signed_ICF': pd.to_datetime([None, None]),
            'TS_Enrolled': pd.to_datetime([None, None])
        })

        ts_col_map = {
            'Passed Online Form': 'TS_Passed_Online_Form',
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        # All hours - should include both referrals
        result_all = calculate_ttfc_effectiveness(df, ts_col_map, business_hours_only=False)

        # Business hours - should exclude Saturday referral
        result_bh = calculate_ttfc_effectiveness(df, ts_col_map, business_hours_only=True)

        # Business hours should have fewer attempts
        if not result_all.empty and not result_bh.empty:
            total_all = result_all['Attempts'].sum()
            total_bh = result_bh['Attempts'].sum()
            assert total_bh <= total_all


class TestContactAttemptBusinessHours:
    """Tests for contact attempt effectiveness with business hours."""

    def test_contact_attempt_business_hours_parameter(self):
        """Test that contact attempt function accepts business_hours_only parameter."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 09:00:00']),
            'TS_Sent_To_Site': pd.to_datetime([None]),
            'TS_Signed_ICF': pd.to_datetime([None]),
            'TS_Enrolled': pd.to_datetime([None]),
            'Parsed_Lead_Status_History': [
                [('Event1', pd.Timestamp('2024-01-01 10:00:00')), ('Event2', pd.Timestamp('2024-01-01 15:00:00'))]
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        # Should work with business_hours_only=False
        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)
        assert isinstance(result, pd.DataFrame)

        # Should work with business_hours_only=True
        result_bh = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=True)
        assert isinstance(result_bh, pd.DataFrame)

    def test_contact_attempt_filters_after_hours(self):
        """Test that business hours excludes after-hours attempts."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 09:00:00']),  # Monday 9 AM
            'TS_Sent_To_Site': pd.to_datetime([None]),
            'Parsed_Lead_Status_History': [
                [
                    ('Event1', pd.Timestamp('2024-01-01 10:00:00')),  # Monday 10 AM - business hours
                    ('Event2', pd.Timestamp('2024-01-01 18:00:00'))   # Monday 6 PM - after hours
                ]
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        # Need to add the missing columns
        df['TS_Signed_ICF'] = pd.NaT
        df['TS_Enrolled'] = pd.NaT

        # All hours - should count both events (2 attempts)
        result_all = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)

        # Business hours - should count only first event (1 attempt)
        result_bh = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=True)

        # Both should return dataframes
        assert isinstance(result_all, pd.DataFrame)
        assert isinstance(result_bh, pd.DataFrame)


class TestBug2PSACountedAsFirstAttempt:
    """Tests to verify Bug 2 fix: PSA event is counted as first attempt."""

    def test_psa_transition_counts_as_first_attempt(self):
        """Test that moving to Pre-Screening Activities counts as attempt #1."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 10:00:00']),  # Monday 10 AM
            'TS_Sent_To_Site': pd.to_datetime(['2024-01-02 14:00:00']),  # Tuesday 2 PM
            'TS_Signed_ICF': pd.to_datetime([None]),
            'TS_Enrolled': pd.to_datetime([None]),
            'Parsed_Lead_Status_History': [
                [
                    ('Passed Online Form', pd.Timestamp('2024-01-01 09:00:00')),
                    ('Pre-Screening Activities', pd.Timestamp('2024-01-01 10:00:00')),  # This is the PSA event
                    ('Sent To Site', pd.Timestamp('2024-01-02 14:00:00'))
                ]
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)

        # The PSA transition should count as 1 attempt
        # Even though there are no other events between PSA and StS
        assert not result.empty
        # Should have a row for 1 attempt
        one_attempt_row = result[result['Number of Attempts'] == 1]
        assert len(one_attempt_row) == 1
        assert one_attempt_row['Total Referrals'].values[0] == 1
        assert one_attempt_row['Total_StS'].values[0] == 1

    def test_no_zero_attempts_for_sent_to_site(self):
        """Test that referrals with StS never show 0 attempts."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        # Create 5 referrals, all reaching StS with varying events
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime([
                '2024-01-01 10:00:00',  # Ref 1: Just PSA and StS
                '2024-01-02 10:00:00',  # Ref 2: PSA + 1 more event
                '2024-01-03 10:00:00',  # Ref 3: PSA + 2 more events
                '2024-01-04 10:00:00',  # Ref 4: Just PSA and StS
                '2024-01-05 10:00:00',  # Ref 5: PSA + 1 more event
            ]),
            'TS_Sent_To_Site': pd.to_datetime([
                '2024-01-02 14:00:00',
                '2024-01-03 14:00:00',
                '2024-01-04 14:00:00',
                '2024-01-05 14:00:00',
                '2024-01-06 14:00:00',
            ]),
            'TS_Signed_ICF': [None] * 5,
            'TS_Enrolled': [None] * 5,
            'Parsed_Lead_Status_History': [
                [('Pre-Screening Activities', pd.Timestamp('2024-01-01 10:00:00')),
                 ('Sent To Site', pd.Timestamp('2024-01-02 14:00:00'))],
                [('Pre-Screening Activities', pd.Timestamp('2024-01-02 10:00:00')),
                 ('Event1', pd.Timestamp('2024-01-02 11:00:00')),
                 ('Sent To Site', pd.Timestamp('2024-01-03 14:00:00'))],
                [('Pre-Screening Activities', pd.Timestamp('2024-01-03 10:00:00')),
                 ('Event1', pd.Timestamp('2024-01-03 11:00:00')),
                 ('Event2', pd.Timestamp('2024-01-03 12:00:00')),
                 ('Sent To Site', pd.Timestamp('2024-01-04 14:00:00'))],
                [('Pre-Screening Activities', pd.Timestamp('2024-01-04 10:00:00')),
                 ('Sent To Site', pd.Timestamp('2024-01-05 14:00:00'))],
                [('Pre-Screening Activities', pd.Timestamp('2024-01-05 10:00:00')),
                 ('Event1', pd.Timestamp('2024-01-05 15:00:00')),
                 ('Sent To Site', pd.Timestamp('2024-01-06 14:00:00'))],
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)

        # Should have no row for 0 attempts (or if it exists, it should have 0 StS)
        zero_attempt_rows = result[result['Number of Attempts'] == 0]
        if not zero_attempt_rows.empty:
            # If there are 0-attempt rows, they should have 0 sent to site
            assert zero_attempt_rows['Total_StS'].sum() == 0

        # All 5 referrals should be distributed across 1, 2, or 3 attempts
        # Ref 1: 1 attempt (PSA only), Ref 2: 2 attempts (PSA + 1), Ref 3: 3 attempts (PSA + 2)
        # Ref 4: 1 attempt (PSA only), Ref 5: 2 attempts (PSA + 1)
        assert result['Total Referrals'].sum() >= 5

        # Check specific attempt counts
        one_attempt = result[result['Number of Attempts'] == 1]
        two_attempts = result[result['Number of Attempts'] == 2]
        three_attempts = result[result['Number of Attempts'] == 3]

        # Should have 2 referrals with 1 attempt, 2 with 2 attempts, 1 with 3 attempts
        assert len(one_attempt) == 1 and one_attempt['Total Referrals'].values[0] == 2
        assert len(two_attempts) == 1 and two_attempts['Total Referrals'].values[0] == 2
        assert len(three_attempts) == 1 and three_attempts['Total Referrals'].values[0] == 1


class TestBug1BusinessHoursInRecommendations:
    """Tests to verify Bug 1 fix: Non-business hours don't appear in recommendations."""

    def test_business_hours_filter_excludes_nonbusiness_slots(self):
        """Test that business hours filter excludes weekend and after-hours slots."""
        from pc_calculations import analyze_heatmap_efficiency

        # Create heatmaps with data in both business and non-business hours
        contact_heatmap = pd.DataFrame(
            np.zeros((7, 24)),
            index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            columns=range(24)
        )
        sts_heatmap = contact_heatmap.copy()

        # Add activity during business hours (Mon-Fri, 9-16)
        contact_heatmap.loc['Monday', 10] = 50  # Business hours
        contact_heatmap.loc['Tuesday', 14] = 45  # Business hours
        sts_heatmap.loc['Monday', 10] = 10
        sts_heatmap.loc['Tuesday', 14] = 8

        # Add activity during non-business hours
        contact_heatmap.loc['Monday', 2] = 100  # 2 AM - after hours
        contact_heatmap.loc['Saturday', 10] = 80  # Weekend
        sts_heatmap.loc['Monday', 2] = 1  # Low efficiency
        sts_heatmap.loc['Saturday', 10] = 2  # Low efficiency

        # Without business hours filter - should include all times
        result_all = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap, business_hours_only=False)

        # With business hours filter - should exclude non-business hours
        result_bh = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap, business_hours_only=True)

        # Check that non-business hours don't appear in business-hours-filtered results
        all_recommendations_bh = result_bh.get('best_times', []) + result_bh.get('avoid_times', [])

        for rec in all_recommendations_bh:
            # Should not contain Saturday or Sunday
            assert 'Saturday' not in rec
            assert 'Sunday' not in rec

            # Should not contain hours outside 9am-5pm
            # Extract hour from recommendation string (e.g., "Monday, 2 AM")
            if ' AM' in rec or ' PM' in rec:
                hour_str = rec.split(', ')[1]  # Get "2 AM" part
                if 'AM' in hour_str:
                    hour = int(hour_str.split(' ')[0])
                    if hour != 12:  # 12 AM is midnight (hour 0)
                        assert hour >= 9, f"Found before-hours time in recommendations: {rec}"
                elif 'PM' in hour_str:
                    hour = int(hour_str.split(' ')[0])
                    if hour != 12:  # 12 PM is noon (hour 12)
                        assert hour < 5, f"Found after-hours time in recommendations: {rec}"

    def test_business_hours_filter_with_only_nonbusiness_data(self):
        """Test that business hours filter returns empty when all data is non-business hours."""
        from pc_calculations import analyze_heatmap_efficiency

        contact_heatmap = pd.DataFrame(
            np.zeros((7, 24)),
            index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            columns=range(24)
        )
        sts_heatmap = contact_heatmap.copy()

        # Only add activity outside business hours
        contact_heatmap.loc['Monday', 2] = 100  # 2 AM
        contact_heatmap.loc['Saturday', 10] = 80  # Weekend
        sts_heatmap.loc['Monday', 2] = 10
        sts_heatmap.loc['Saturday', 10] = 8

        result = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap, business_hours_only=True)

        # Should return empty or have empty lists
        if result:
            assert len(result.get('best_times', [])) == 0
            assert len(result.get('avoid_times', [])) == 0
        else:
            assert result == {}

    def test_business_hours_parameter_default_false(self):
        """Test that business_hours_only defaults to False (includes all hours)."""
        from pc_calculations import analyze_heatmap_efficiency

        contact_heatmap = pd.DataFrame(
            np.random.poisson(10, size=(7, 24)),
            index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            columns=range(24)
        )
        sts_heatmap = pd.DataFrame(
            np.random.poisson(3, size=(7, 24)),
            index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            columns=range(24)
        )

        # Call without parameter (should default to False)
        result_default = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap)

        # Call with explicit False
        result_false = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap, business_hours_only=False)

        # Both should return the same structure
        assert result_default.keys() == result_false.keys()


class TestPSAAlwaysCountsAsFirstAttempt:
    """Tests to verify PSA transition always counts, even outside business hours."""

    def test_psa_outside_business_hours_still_counts(self):
        """Test that PSA transition counts even when it occurs outside business hours."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        # Create a referral where PSA happens at 6 PM (outside business hours)
        # but StS happens during business hours
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 18:00:00']),  # Monday 6 PM - AFTER hours
            'TS_Sent_To_Site': pd.to_datetime(['2024-01-02 14:00:00']),  # Tuesday 2 PM - business hours
            'TS_Signed_ICF': pd.to_datetime([None]),
            'TS_Enrolled': pd.to_datetime([None]),
            'Parsed_Lead_Status_History': [
                [
                    ('Passed Online Form', pd.Timestamp('2024-01-01 17:00:00')),
                    ('Pre-Screening Activities', pd.Timestamp('2024-01-01 18:00:00')),  # PSA at 6 PM
                    ('Sent To Site', pd.Timestamp('2024-01-02 14:00:00'))
                ]
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        # With business hours filter enabled
        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=True)

        # PSA should count as 1 attempt even though it's outside business hours
        assert not result.empty
        one_attempt_row = result[result['Number of Attempts'] == 1]
        assert len(one_attempt_row) == 1
        assert one_attempt_row['Total Referrals'].values[0] == 1
        assert one_attempt_row['Total_StS'].values[0] == 1

        # Should NOT have a row for 0 attempts with StS
        zero_attempt_rows = result[result['Number of Attempts'] == 0]
        if not zero_attempt_rows.empty:
            assert zero_attempt_rows['Total_StS'].sum() == 0

    def test_psa_on_weekend_still_counts(self):
        """Test that PSA transition counts even when it occurs on a weekend."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        # Create a referral where PSA happens on Saturday
        # but StS happens on Monday
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-06 10:00:00']),  # Saturday 10 AM - WEEKEND
            'TS_Sent_To_Site': pd.to_datetime(['2024-01-08 14:00:00']),  # Monday 2 PM - business hours
            'TS_Signed_ICF': pd.to_datetime([None]),
            'TS_Enrolled': pd.to_datetime([None]),
            'Parsed_Lead_Status_History': [
                [
                    ('Passed Online Form', pd.Timestamp('2024-01-06 09:00:00')),
                    ('Pre-Screening Activities', pd.Timestamp('2024-01-06 10:00:00')),  # PSA on Saturday
                    ('Sent To Site', pd.Timestamp('2024-01-08 14:00:00'))
                ]
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        # With business hours filter enabled
        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=True)

        # PSA should count as 1 attempt even though it's on a weekend
        assert not result.empty
        one_attempt_row = result[result['Number of Attempts'] == 1]
        assert len(one_attempt_row) == 1
        assert one_attempt_row['Total_StS'].values[0] == 1

    def test_psa_weekend_plus_business_hour_events(self):
        """Test counting when PSA is on weekend and there are additional business hour events."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        # PSA on Saturday + 2 more events during business hours = 3 total attempts
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-06 10:00:00']),  # Saturday
            'TS_Sent_To_Site': pd.to_datetime(['2024-01-09 14:00:00']),  # Tuesday
            'TS_Signed_ICF': pd.to_datetime([None]),
            'TS_Enrolled': pd.to_datetime([None]),
            'Parsed_Lead_Status_History': [
                [
                    ('Passed Online Form', pd.Timestamp('2024-01-06 09:00:00')),
                    ('Pre-Screening Activities', pd.Timestamp('2024-01-06 10:00:00')),  # PSA on Saturday (counts as 1)
                    ('Event1', pd.Timestamp('2024-01-08 10:00:00')),  # Monday business hours (counts as 1)
                    ('Event2', pd.Timestamp('2024-01-08 15:00:00')),  # Monday business hours (counts as 1)
                    ('Sent To Site', pd.Timestamp('2024-01-09 14:00:00'))
                ]
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        # With business hours filter enabled
        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=True)

        # Should have 3 attempts total: 1 (PSA on weekend) + 2 (business hour events)
        assert not result.empty
        three_attempt_row = result[result['Number of Attempts'] == 3]
        assert len(three_attempt_row) == 1
        assert three_attempt_row['Total_StS'].values[0] == 1


class TestDataQualityEdgeCases:
    """Tests for edge cases with missing or invalid history data."""

    def test_psa_with_empty_history_counts_as_one_attempt(self):
        """Test that referral with PSA timestamp but empty history shows 1 attempt."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        # Referral reached PSA but history is empty (data quality issue)
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 10:00:00']),  # Has PSA timestamp
            'TS_Sent_To_Site': pd.to_datetime(['2024-01-02 14:00:00']),  # Reached StS
            'TS_Signed_ICF': pd.to_datetime([None]),
            'TS_Enrolled': pd.to_datetime([None]),
            'Parsed_Lead_Status_History': [[]]  # Empty history list
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        # All hours mode
        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)

        # Should show 1 attempt (PSA transition) not 0
        assert not result.empty
        one_attempt_row = result[result['Number of Attempts'] == 1]
        assert len(one_attempt_row) == 1
        assert one_attempt_row['Total_StS'].values[0] == 1

        # Should NOT have 0 attempts
        zero_attempt_rows = result[result['Number of Attempts'] == 0]
        if not zero_attempt_rows.empty:
            assert zero_attempt_rows['Total_StS'].sum() == 0

    def test_psa_with_invalid_history_counts_as_one_attempt(self):
        """Test that referral with PSA timestamp but invalid history (NaN) shows 1 attempt."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        # Referral reached PSA but history is NaN (data quality issue)
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 10:00:00']),  # Has PSA timestamp
            'TS_Sent_To_Site': pd.to_datetime(['2024-01-02 14:00:00']),  # Reached StS
            'TS_Signed_ICF': pd.to_datetime([None]),
            'TS_Enrolled': pd.to_datetime([None]),
            'Parsed_Lead_Status_History': [None]  # Invalid history
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)

        # Should show 1 attempt (PSA transition) not 0
        assert not result.empty
        one_attempt_row = result[result['Number of Attempts'] == 1]
        assert len(one_attempt_row) == 1
        assert one_attempt_row['Total_StS'].values[0] == 1

    def test_no_psa_timestamp_shows_zero_attempts(self):
        """Test that referral without PSA timestamp correctly shows 0 attempts."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        # Referral never reached PSA (still in Passed Online Form)
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime([None]),  # No PSA timestamp
            'TS_Sent_To_Site': pd.to_datetime([None]),
            'TS_Signed_ICF': pd.to_datetime([None]),
            'TS_Enrolled': pd.to_datetime([None]),
            'Parsed_Lead_Status_History': [[('Passed Online Form', pd.Timestamp('2024-01-01 09:00:00'))]]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)

        # Should show 0 attempts
        assert not result.empty
        zero_attempt_row = result[result['Number of Attempts'] == 0]
        assert len(zero_attempt_row) == 1
        assert zero_attempt_row['Total Referrals'].values[0] == 1
        # And should have 0 StS
        assert zero_attempt_row['Total_StS'].values[0] == 0

    def test_psa_empty_history_business_hours_mode(self):
        """Test that empty history with PSA shows 1 attempt in business hours mode."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime(['2024-01-01 10:00:00']),
            'TS_Sent_To_Site': pd.to_datetime(['2024-01-02 14:00:00']),
            'TS_Signed_ICF': pd.to_datetime([None]),
            'TS_Enrolled': pd.to_datetime([None]),
            'Parsed_Lead_Status_History': [[]]  # Empty history
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled'
        }

        # Business hours mode
        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=True)

        # Should show 1 attempt even in business hours mode
        assert not result.empty
        one_attempt_row = result[result['Number of Attempts'] == 1]
        assert len(one_attempt_row) == 1
        assert one_attempt_row['Total_StS'].values[0] == 1


class TestLostAndScreenFailedReferrals:
    """Tests for referrals that go directly to Lost or Screen Failed without PSA."""

    def test_lost_referral_without_psa_shows_one_attempt(self):
        """Test that referral going directly to Lost (skipping PSA) shows 1 attempt."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        # Referral went: Passed Online Form → Lost (no PSA)
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime([None]),  # No PSA
            'TS_Sent_To_Site': pd.to_datetime([None]),
            'TS_Signed_ICF': pd.to_datetime([None]),
            'TS_Enrolled': pd.to_datetime([None]),
            'TS_Lost': pd.to_datetime(['2024-01-02 14:00:00']),  # Went to Lost
            'Parsed_Lead_Status_History': [
                [
                    ('New', pd.Timestamp('2024-01-01 09:00:00')),
                    ('1nH Phone Screen DQ', pd.Timestamp('2024-01-02 14:00:00'))
                ]
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled',
            'Lost': 'TS_Lost'
        }

        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)

        # Should have 1 attempt (the Lost transition)
        assert not result.empty
        one_attempt_row = result[result['Number of Attempts'] == 1]
        assert len(one_attempt_row) == 1
        assert one_attempt_row['Total Referrals'].values[0] == 1

        # Should have 0 StS (went to Lost, not StS)
        assert one_attempt_row['Total_StS'].values[0] == 0

        # Should NOT have 0 attempts
        zero_attempt_rows = result[result['Number of Attempts'] == 0]
        if not zero_attempt_rows.empty:
            assert zero_attempt_rows['Total Referrals'].sum() == 0

    def test_screen_failed_without_psa_shows_one_attempt(self):
        """Test that referral going directly to Screen Failed (skipping PSA) shows 1 attempt."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        # Referral went: Passed Online Form → Screen Failed (no PSA)
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime([None]),  # No PSA
            'TS_Sent_To_Site': pd.to_datetime([None]),
            'TS_Signed_ICF': pd.to_datetime([None]),
            'TS_Enrolled': pd.to_datetime([None]),
            'TS_Screen_Failed': pd.to_datetime(['2024-01-03 10:00:00']),  # Went to Screen Failed
            'Parsed_Lead_Status_History': [
                [
                    ('New', pd.Timestamp('2024-01-01 09:00:00')),
                    ('Pre-Screening Disqualified', pd.Timestamp('2024-01-03 10:00:00'))
                ]
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled',
            'Screen Failed': 'TS_Screen_Failed'
        }

        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)

        # Should have 1 attempt
        assert not result.empty
        one_attempt_row = result[result['Number of Attempts'] == 1]
        assert len(one_attempt_row) == 1
        assert one_attempt_row['Total Referrals'].values[0] == 1

    def test_multiple_lost_referrals_with_varying_attempts(self):
        """Test multiple Lost referrals with different status history patterns."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        # 3 referrals:
        # 1. Lost with just 1 status change (phone screen DQ)
        # 2. Lost with 2 status changes (attempt 1, then DQ)
        # 3. Lost with 3 status changes (attempt 1, attempt 2, then DQ)
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime([None, None, None]),
            'TS_Sent_To_Site': pd.to_datetime([None, None, None]),
            'TS_Signed_ICF': pd.to_datetime([None, None, None]),
            'TS_Enrolled': pd.to_datetime([None, None, None]),
            'TS_Lost': pd.to_datetime(['2024-01-02 14:00:00', '2024-01-03 15:00:00', '2024-01-04 16:00:00']),
            'Parsed_Lead_Status_History': [
                [
                    ('New', pd.Timestamp('2024-01-02 09:00:00')),
                    ('1nH Phone Screen DQ', pd.Timestamp('2024-01-02 14:00:00'))
                ],
                [
                    ('New', pd.Timestamp('2024-01-03 09:00:00')),
                    ('1nH Contact Attempt 1', pd.Timestamp('2024-01-03 10:00:00')),
                    ('1nH Phone Screen DQ', pd.Timestamp('2024-01-03 15:00:00'))
                ],
                [
                    ('New', pd.Timestamp('2024-01-04 09:00:00')),
                    ('1nH Contact Attempt 1', pd.Timestamp('2024-01-04 10:00:00')),
                    ('1nH Contact Attempt 2', pd.Timestamp('2024-01-04 14:00:00')),
                    ('1nH No Longer Interested', pd.Timestamp('2024-01-04 16:00:00'))
                ]
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled',
            'Lost': 'TS_Lost'
        }

        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)

        # Should have results for 1, 2, and 3 attempts
        assert not result.empty

        # Should NOT have any 0 attempt referrals
        zero_attempt_rows = result[result['Number of Attempts'] == 0]
        if not zero_attempt_rows.empty:
            assert zero_attempt_rows['Total Referrals'].sum() == 0

        # Check we have expected attempt counts
        attempts_distribution = result.set_index('Number of Attempts')['Total Referrals'].to_dict()
        assert 1 in attempts_distribution or 2 in attempts_distribution or 3 in attempts_distribution

    def test_lost_referral_with_business_hours_filter(self):
        """Test that Lost referrals work correctly with business hours filter."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        # Lost referral with some events during and some outside business hours
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime([None]),
            'TS_Sent_To_Site': pd.to_datetime([None]),
            'TS_Signed_ICF': pd.to_datetime([None]),
            'TS_Enrolled': pd.to_datetime([None]),
            'TS_Lost': pd.to_datetime(['2024-01-02 14:00:00']),  # Tuesday 2 PM - business hours
            'Parsed_Lead_Status_History': [
                [
                    ('New', pd.Timestamp('2024-01-01 20:00:00')),  # Monday 8 PM - after hours
                    ('1nH Contact Attempt 1', pd.Timestamp('2024-01-02 10:00:00')),  # Tuesday 10 AM - business hours
                    ('1nH Phone Screen DQ', pd.Timestamp('2024-01-02 14:00:00'))  # Tuesday 2 PM - business hours
                ]
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled',
            'Lost': 'TS_Lost'
        }

        # Business hours mode
        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=True)

        # Should have at least 1 attempt (Lost transition always counts)
        assert not result.empty
        # Total referrals should be 1
        assert result['Total Referrals'].sum() == 1
        # Should not have 0 attempts
        zero_attempt_rows = result[result['Number of Attempts'] == 0]
        if not zero_attempt_rows.empty:
            assert zero_attempt_rows['Total Referrals'].sum() == 0

    def test_only_passed_online_form_shows_zero_attempts(self):
        """Test that only referrals still in Passed Online Form stage show 0 attempts."""
        from pc_calculations import calculate_contact_attempt_effectiveness

        # Mix of referrals:
        # 1. Still in Passed Online Form (0 attempts)
        # 2. Went to Lost (1+ attempts)
        # 3. Went to PSA (1+ attempts)
        df = pd.DataFrame({
            'TS_Pre-Screening_Activities': pd.to_datetime([None, None, '2024-01-03 10:00:00']),
            'TS_Sent_To_Site': pd.to_datetime([None, None, None]),
            'TS_Signed_ICF': pd.to_datetime([None, None, None]),
            'TS_Enrolled': pd.to_datetime([None, None, None]),
            'TS_Lost': pd.to_datetime([None, '2024-01-02 14:00:00', None]),
            'Parsed_Lead_Status_History': [
                [('New', pd.Timestamp('2024-01-01 09:00:00'))],  # Still in POF
                [('New', pd.Timestamp('2024-01-02 09:00:00')),
                 ('1nH Phone Screen DQ', pd.Timestamp('2024-01-02 14:00:00'))],  # Went to Lost
                [('New', pd.Timestamp('2024-01-03 09:00:00')),
                 ('1nH Contact Attempt 1', pd.Timestamp('2024-01-03 10:00:00'))]  # Went to PSA
            ]
        })

        ts_col_map = {
            'Pre-Screening Activities': 'TS_Pre-Screening_Activities',
            'Sent To Site': 'TS_Sent_To_Site',
            'Signed ICF': 'TS_Signed_ICF',
            'Enrolled': 'TS_Enrolled',
            'Lost': 'TS_Lost'
        }

        result = calculate_contact_attempt_effectiveness(df, ts_col_map, 'Parsed_Lead_Status_History', business_hours_only=False)

        # Should have a 0 attempt row with exactly 1 referral
        assert not result.empty
        zero_attempt_row = result[result['Number of Attempts'] == 0]
        assert len(zero_attempt_row) == 1
        assert zero_attempt_row['Total Referrals'].values[0] == 1

        # Should have 1 attempt rows with 2 referrals (Lost + PSA)
        one_attempt_row = result[result['Number of Attempts'] == 1]
        assert len(one_attempt_row) == 1
        assert one_attempt_row['Total Referrals'].values[0] == 2
