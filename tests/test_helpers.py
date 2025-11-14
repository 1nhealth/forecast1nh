"""Tests for helper functions."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from helpers import format_days_to_dhm, calculate_avg_lag_generic, calculate_business_hours_between


class TestFormatDaysToDhm:
    """Tests for format_days_to_dhm function."""

    def test_zero_days(self):
        """Test formatting zero days."""
        result = format_days_to_dhm(0)
        assert result == "0 d 0 h 0 m"

    def test_whole_days(self):
        """Test formatting whole days."""
        result = format_days_to_dhm(3.0)
        assert result == "3 d 0 h 0 m"

    def test_days_and_hours(self):
        """Test formatting days and hours."""
        result = format_days_to_dhm(1.5)  # 1.5 days = 1 day 12 hours
        assert result == "1 d 12 h 0 m"

    def test_days_hours_minutes(self):
        """Test formatting days, hours, and minutes."""
        # 1.0625 days = 1 day 1.5 hours = 1 day 1 hour 30 minutes
        result = format_days_to_dhm(1.0625)
        assert result == "1 d 1 h 30 m"

    def test_only_hours(self):
        """Test formatting when less than a day."""
        result = format_days_to_dhm(0.5)  # 0.5 days = 12 hours
        assert result == "0 d 12 h 0 m"

    def test_only_minutes(self):
        """Test formatting when less than an hour."""
        result = format_days_to_dhm(0.02083333)  # ~30 minutes
        assert result == "0 d 0 h 29 m"  # Rounding to nearest minute

    def test_negative_days(self):
        """Test formatting negative days."""
        result = format_days_to_dhm(-1.5)
        assert result == "N/A"  # Returns N/A for negatives


class TestCalculateAvgLagGeneric:
    """Tests for calculate_avg_lag_generic function."""

    def test_basic_lag_calculation(self):
        """Test basic lag calculation between two timestamps."""
        df = pd.DataFrame({
            'start': pd.to_datetime(['2024-01-01 09:00:00', '2024-01-02 09:00:00']),
            'end': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-02 11:00:00'])
        })
        result = calculate_avg_lag_generic(df, 'start', 'end')
        # Average of 1 hour and 2 hours = 1.5 hours = 0.0625 days
        assert abs(result - 0.0625) < 0.001

    def test_with_null_values(self):
        """Test lag calculation with null values."""
        df = pd.DataFrame({
            'start': pd.to_datetime(['2024-01-01 09:00:00', '2024-01-02 09:00:00', None]),
            'end': pd.to_datetime(['2024-01-01 10:00:00', None, '2024-01-03 09:00:00'])
        })
        result = calculate_avg_lag_generic(df, 'start', 'end')
        # Only first row is valid: 1 hour = 0.041667 days
        assert abs(result - 0.041667) < 0.001

    def test_all_null_values(self):
        """Test lag calculation with all null values."""
        df = pd.DataFrame({
            'start': [None, None],
            'end': [None, None]
        })
        result = calculate_avg_lag_generic(df, 'start', 'end')
        assert pd.isna(result)  # Returns NaN for all nulls

    def test_negative_lag_excluded(self):
        """Test that negative lags are excluded."""
        df = pd.DataFrame({
            'start': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-02 09:00:00']),
            'end': pd.to_datetime(['2024-01-01 09:00:00', '2024-01-02 11:00:00'])  # First is negative
        })
        result = calculate_avg_lag_generic(df, 'start', 'end')
        # Only second row: 2 hours = 0.083333 days
        assert abs(result - 0.083333) < 0.001

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame({'start': [], 'end': []})
        result = calculate_avg_lag_generic(df, 'start', 'end')
        assert pd.isna(result)  # Returns NaN for empty data


class TestCalculateBusinessHoursBetween:
    """Tests for calculate_business_hours_between function."""

    def test_same_timestamp(self):
        """Test with same start and end timestamp."""
        start = pd.Timestamp('2024-01-01 10:00:00')  # Monday 10 AM
        end = pd.Timestamp('2024-01-01 10:00:00')
        result = calculate_business_hours_between(start, end)
        assert result == 0

    def test_negative_timespan(self):
        """Test with end before start."""
        start = pd.Timestamp('2024-01-01 14:00:00')  # Monday 2 PM
        end = pd.Timestamp('2024-01-01 10:00:00')    # Monday 10 AM
        result = calculate_business_hours_between(start, end)
        assert result == 0

    def test_invalid_inputs(self):
        """Test with invalid/NaN inputs."""
        start = pd.Timestamp('2024-01-01 10:00:00')

        # Test with NaN end
        result = calculate_business_hours_between(start, pd.NaT)
        assert pd.isna(result)

        # Test with NaN start
        result = calculate_business_hours_between(pd.NaT, start)
        assert pd.isna(result)

        # Test with both NaN
        result = calculate_business_hours_between(pd.NaT, pd.NaT)
        assert pd.isna(result)

    def test_within_same_business_day(self):
        """Test calculation within same business day."""
        # Monday 10 AM to Monday 2 PM = 4 hours = 4/24 days
        start = pd.Timestamp('2024-01-01 10:00:00')  # Monday 10 AM
        end = pd.Timestamp('2024-01-01 14:00:00')    # Monday 2 PM
        result = calculate_business_hours_between(start, end)
        expected = 4 / 24  # 4 hours in days
        assert abs(result - expected) < 0.001

    def test_spanning_overnight_business_days(self):
        """Test spanning overnight between business days."""
        # Monday 4 PM to Tuesday 10 AM = 1 hour Mon + 1 hour Tue = 2 hours
        start = pd.Timestamp('2024-01-01 16:00:00')  # Monday 4 PM
        end = pd.Timestamp('2024-01-02 10:00:00')    # Tuesday 10 AM
        result = calculate_business_hours_between(start, end)
        expected = 2 / 24  # 2 business hours in days
        assert abs(result - expected) < 0.001

    def test_spanning_weekend(self):
        """Test spanning a weekend."""
        # Friday 4 PM to Monday 10 AM = 1 hour Fri + 1 hour Mon = 2 hours
        start = pd.Timestamp('2024-01-05 16:00:00')  # Friday 4 PM
        end = pd.Timestamp('2024-01-08 10:00:00')    # Monday 10 AM
        result = calculate_business_hours_between(start, end)
        expected = 2 / 24  # 2 business hours in days
        assert abs(result - expected) < 0.001

    def test_start_before_business_hours(self):
        """Test starting before business hours."""
        # Monday 8 AM to Monday 11 AM = 2 hours (9-11, 8-9 doesn't count)
        start = pd.Timestamp('2024-01-01 08:00:00')  # Monday 8 AM
        end = pd.Timestamp('2024-01-01 11:00:00')    # Monday 11 AM
        result = calculate_business_hours_between(start, end)
        expected = 2 / 24  # 2 business hours in days
        assert abs(result - expected) < 0.001

    def test_end_after_business_hours(self):
        """Test ending after business hours."""
        # Monday 4 PM to Monday 6 PM = 1 hour (4-5, 5-6 doesn't count)
        start = pd.Timestamp('2024-01-01 16:00:00')  # Monday 4 PM
        end = pd.Timestamp('2024-01-01 18:00:00')    # Monday 6 PM
        result = calculate_business_hours_between(start, end)
        expected = 1 / 24  # 1 business hour in days
        assert abs(result - expected) < 0.001

    def test_fully_outside_business_hours(self):
        """Test time span fully outside business hours."""
        # Monday 6 PM to Monday 8 PM = 0 hours (all outside business hours)
        start = pd.Timestamp('2024-01-01 18:00:00')  # Monday 6 PM
        end = pd.Timestamp('2024-01-01 20:00:00')    # Monday 8 PM
        result = calculate_business_hours_between(start, end)
        assert result == 0

    def test_weekend_only(self):
        """Test time span spanning weekend only."""
        # Friday 5 PM to Monday 9 AM = 0 hours (all outside business hours)
        start = pd.Timestamp('2024-01-05 17:00:00')  # Friday 5 PM
        end = pd.Timestamp('2024-01-08 09:00:00')    # Monday 9 AM
        result = calculate_business_hours_between(start, end)
        assert result == 0

    def test_full_business_day(self):
        """Test a full business day (9 AM to 5 PM)."""
        # Monday 9 AM to Monday 5 PM = 8 hours
        start = pd.Timestamp('2024-01-01 09:00:00')  # Monday 9 AM
        end = pd.Timestamp('2024-01-01 17:00:00')    # Monday 5 PM
        result = calculate_business_hours_between(start, end)
        expected = 8 / 24  # 8 business hours in days
        assert abs(result - expected) < 0.001

    def test_multiple_full_business_days(self):
        """Test spanning multiple full business days."""
        # Monday 9 AM to Wednesday 5 PM = 8 + 8 + 8 = 24 hours
        start = pd.Timestamp('2024-01-01 09:00:00')  # Monday 9 AM
        end = pd.Timestamp('2024-01-03 17:00:00')    # Wednesday 5 PM
        result = calculate_business_hours_between(start, end)
        expected = 24 / 24  # 24 business hours = 1 day
        assert abs(result - expected) < 0.001

    def test_spanning_multiple_weeks(self):
        """Test spanning multiple weeks including weekends."""
        # Monday Jan 1 9 AM to Monday Jan 8 5 PM = 5 full days + partial = 40 hours + 8 = 48 hours
        start = pd.Timestamp('2024-01-01 09:00:00')  # Monday 9 AM
        end = pd.Timestamp('2024-01-08 17:00:00')    # Next Monday 5 PM
        result = calculate_business_hours_between(start, end)
        # Mon, Tue, Wed, Thu, Fri = 40 hours, then Mon again = 8 hours, total 48 hours
        expected = 48 / 24  # 48 business hours = 2 days
        assert abs(result - expected) < 0.001


class TestCalculateAvgLagGenericBusinessHours:
    """Tests for calculate_avg_lag_generic with business_hours_only parameter."""

    def test_business_hours_basic(self):
        """Test basic business hours calculation."""
        df = pd.DataFrame({
            'start': pd.to_datetime([
                '2024-01-01 10:00:00',  # Monday 10 AM
                '2024-01-02 14:00:00'   # Tuesday 2 PM
            ]),
            'end': pd.to_datetime([
                '2024-01-01 14:00:00',  # Monday 2 PM (4 hours)
                '2024-01-02 16:00:00'   # Tuesday 4 PM (2 hours)
            ])
        })
        result = calculate_avg_lag_generic(df, 'start', 'end', business_hours_only=True)
        # Average of 4 hours and 2 hours = 3 hours = 3/24 = 0.125 days
        expected = 3 / 24
        assert abs(result - expected) < 0.001

    def test_business_hours_vs_calendar_time(self):
        """Test that business hours differs from calendar time when spanning weekend."""
        # Friday 4 PM to Monday 10 AM
        df = pd.DataFrame({
            'start': pd.to_datetime(['2024-01-05 16:00:00']),  # Friday 4 PM
            'end': pd.to_datetime(['2024-01-08 10:00:00'])      # Monday 10 AM
        })

        # Calendar time: ~65 hours = ~2.7 days
        calendar_result = calculate_avg_lag_generic(df, 'start', 'end', business_hours_only=False)

        # Business hours: 1 hour Fri + 1 hour Mon = 2 hours = 2/24 days
        business_result = calculate_avg_lag_generic(df, 'start', 'end', business_hours_only=True)

        # Business hours should be much less than calendar time
        assert business_result < calendar_result
        assert abs(business_result - (2/24)) < 0.001

    def test_business_hours_with_nulls(self):
        """Test business hours calculation with null values."""
        df = pd.DataFrame({
            'start': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-02 14:00:00',
                None
            ]),
            'end': pd.to_datetime([
                '2024-01-01 14:00:00',
                None,
                '2024-01-03 16:00:00'
            ])
        })
        result = calculate_avg_lag_generic(df, 'start', 'end', business_hours_only=True)
        # Only first row is valid: 4 hours = 4/24 days
        expected = 4 / 24
        assert abs(result - expected) < 0.001

    def test_business_hours_negative_excluded(self):
        """Test that negative time spans are excluded in business hours mode."""
        df = pd.DataFrame({
            'start': pd.to_datetime([
                '2024-01-01 14:00:00',  # Monday 2 PM
                '2024-01-02 10:00:00'   # Tuesday 10 AM
            ]),
            'end': pd.to_datetime([
                '2024-01-01 10:00:00',  # Monday 10 AM (negative)
                '2024-01-02 14:00:00'   # Tuesday 2 PM (4 hours)
            ])
        })
        result = calculate_avg_lag_generic(df, 'start', 'end', business_hours_only=True)
        # Only second row: 4 hours = 4/24 days
        expected = 4 / 24
        assert abs(result - expected) < 0.001

    def test_business_hours_all_outside(self):
        """Test when all time spans are outside business hours."""
        df = pd.DataFrame({
            'start': pd.to_datetime([
                '2024-01-01 18:00:00',  # Monday 6 PM
                '2024-01-01 20:00:00'   # Monday 8 PM
            ]),
            'end': pd.to_datetime([
                '2024-01-01 20:00:00',  # Monday 8 PM
                '2024-01-01 22:00:00'   # Monday 10 PM
            ])
        })
        result = calculate_avg_lag_generic(df, 'start', 'end', business_hours_only=True)
        # All times outside business hours = 0 average
        assert result == 0

    def test_business_hours_empty_df(self):
        """Test business hours calculation with empty dataframe."""
        df = pd.DataFrame({'start': [], 'end': []})
        result = calculate_avg_lag_generic(df, 'start', 'end', business_hours_only=True)
        assert pd.isna(result)

    def test_default_parameter_is_false(self):
        """Test that business_hours_only defaults to False."""
        df = pd.DataFrame({
            'start': pd.to_datetime(['2024-01-05 16:00:00']),  # Friday 4 PM
            'end': pd.to_datetime(['2024-01-08 10:00:00'])      # Monday 10 AM
        })

        # Default should be calendar time
        result_default = calculate_avg_lag_generic(df, 'start', 'end')
        result_explicit_false = calculate_avg_lag_generic(df, 'start', 'end', business_hours_only=False)

        # Both should be equal (calendar time)
        assert result_default == result_explicit_false


# Business hours tests
class TestBusinessHours:
    """Tests for business hours functions."""

    def test_is_business_hours_weekday_valid(self, business_hours_timestamps):
        """Test business hours detection for valid weekday hours."""
        from helpers import is_business_hours

        # Monday 9 AM - should be business hours
        assert is_business_hours(business_hours_timestamps['monday_9am']) == True

        # Friday 2 PM - should be business hours
        assert is_business_hours(business_hours_timestamps['friday_2pm']) == True

    def test_is_business_hours_before_hours(self, business_hours_timestamps):
        """Test business hours detection for before hours."""
        from helpers import is_business_hours

        # Monday 8 AM - before business hours
        assert is_business_hours(business_hours_timestamps['monday_8am']) == False

    def test_is_business_hours_at_boundary(self, business_hours_timestamps):
        """Test business hours detection at 5 PM boundary."""
        from helpers import is_business_hours

        # Monday 5 PM - should be False (end of business hours)
        assert is_business_hours(business_hours_timestamps['monday_5pm']) == False

    def test_is_business_hours_after_hours(self, business_hours_timestamps):
        """Test business hours detection for after hours."""
        from helpers import is_business_hours

        # Monday 6 PM - after business hours
        assert is_business_hours(business_hours_timestamps['monday_6pm']) == False

    def test_is_business_hours_weekend(self, business_hours_timestamps):
        """Test business hours detection for weekends."""
        from helpers import is_business_hours

        # Saturday 10 AM - weekend, should be False
        assert is_business_hours(business_hours_timestamps['saturday_10am']) == False

        # Sunday 10 AM - weekend, should be False
        assert is_business_hours(business_hours_timestamps['sunday_10am']) == False

    def test_is_business_hours_null(self):
        """Test business hours detection with null timestamp."""
        from helpers import is_business_hours

        assert is_business_hours(pd.NaT) == False
        assert is_business_hours(None) == False

    def test_filter_business_hours_only(self):
        """Test filtering dataframe to business hours only."""
        from helpers import filter_business_hours_only

        df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 09:00:00',  # Monday 9 AM - business hours
                '2024-01-01 08:00:00',  # Monday 8 AM - before hours
                '2024-01-01 18:00:00',  # Monday 6 PM - after hours
                '2024-01-06 10:00:00',  # Saturday 10 AM - weekend
                '2024-01-02 14:00:00',  # Tuesday 2 PM - business hours
            ]),
            'value': [1, 2, 3, 4, 5]
        })

        filtered = filter_business_hours_only(df, 'timestamp')

        # Should only have 2 rows (Monday 9 AM and Tuesday 2 PM)
        assert len(filtered) == 2
        assert filtered['value'].tolist() == [1, 5]

    def test_filter_business_hours_empty_df(self):
        """Test filtering empty dataframe."""
        from helpers import filter_business_hours_only

        df = pd.DataFrame({'timestamp': [], 'value': []})
        filtered = filter_business_hours_only(df, 'timestamp')

        assert filtered.empty

    def test_filter_business_hours_invalid_column(self):
        """Test filtering with invalid column name."""
        from helpers import filter_business_hours_only

        df = pd.DataFrame({'timestamp': pd.to_datetime(['2024-01-01 09:00:00']), 'value': [1]})
        filtered = filter_business_hours_only(df, 'nonexistent_column')

        # Should return original dataframe if column doesn't exist
        assert len(filtered) == 1
