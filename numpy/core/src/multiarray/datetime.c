/*
 * This file implements core functionality for NumPy datetime.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <datetime.h>

#include <time.h>

#define NPY_NO_DEPRECATED_API
#define _MULTIARRAYMODULE
#include <numpy/arrayobject.h>

#include "npy_config.h"
#include "numpy/npy_3kcompat.h"

#include "numpy/arrayscalars.h"
#include "methods.h"
#include "_datetime.h"
#include "datetime_strings.h"

/*
 * Imports the PyDateTime functions so we can create these objects.
 * This is called during module initialization
 */
NPY_NO_EXPORT void
numpy_pydatetime_import()
{
    PyDateTime_IMPORT;
}

/* Exported as DATETIMEUNITS in multiarraymodule.c */
NPY_NO_EXPORT char *_datetime_strings[NPY_DATETIME_NUMUNITS] = {
    NPY_STR_Y,
    NPY_STR_M,
    NPY_STR_W,
    NPY_STR_B,
    NPY_STR_D,
    NPY_STR_h,
    NPY_STR_m,
    NPY_STR_s,
    NPY_STR_ms,
    NPY_STR_us,
    NPY_STR_ns,
    NPY_STR_ps,
    NPY_STR_fs,
    NPY_STR_as,
    "generic"
};

/* Gets the day of the week for a datetime64[D] value */
static int
get_day_of_week(npy_datetime date)
{
    int day_of_week;

    /* Get the day of the week for 'date' (1970-01-05 is Monday) */
    day_of_week = (int)((date - 4) % 7);
    if (day_of_week < 0) {
        day_of_week += 7;
    }

#ifdef DATETIME_DEBUG
    fprintf(stderr, "day_of_week: %lld -> %d\n",
            (long long)date, day_of_week);
#endif
    return day_of_week;
}

/* Days per month, regular year and leap year */
NPY_NO_EXPORT int _days_per_month_table[2][12] = {
    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

/*
 * Returns 1 if the given year is a leap year, 0 otherwise.
 */
NPY_NO_EXPORT int
is_leapyear(npy_int64 year)
{
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 ||
            (year % 400) == 0);
}

/*
 * Calculates the days offset from the 1970 epoch.
 */
NPY_NO_EXPORT npy_int64
get_datetimestruct_days(const npy_datetimestruct *dts)
{
    int i, month;
    npy_int64 year, days = 0;
    int *month_lengths;

    year = dts->year - 1970;
    days = year * 365;

    /* Adjust for leap years */
    if (days >= 0) {
        /*
         * 1968 is the closest leap year before 1970.
         * Exclude the current year, so add 1.
         */
        year += 1;
        /* Add one day for each 4 years */
        days += year / 4;
        /* 1900 is the closest previous year divisible by 100 */
        year += 68;
        /* Subtract one day for each 100 years */
        days -= year / 100;
        /* 1600 is the closest previous year divisible by 400 */
        year += 300;
        /* Add one day for each 400 years */
        days += year / 400;
    }
    else {
        /*
         * 1972 is the closest later year after 1970.
         * Include the current year, so subtract 2.
         */
        year -= 2;
        /* Subtract one day for each 4 years */
        days += year / 4;
        /* 2000 is the closest later year divisible by 100 */
        year -= 28;
        /* Add one day for each 100 years */
        days -= year / 100;
        /* 2000 is also the closest later year divisible by 400 */
        /* Subtract one day for each 400 years */
        days += year / 400;
    }

    month_lengths = _days_per_month_table[is_leapyear(dts->year)];
    month = dts->month - 1;

    /* Add the months */
    for (i = 0; i < month; ++i) {
        days += month_lengths[i];
    }

    /* Add the days */
    days += dts->day - 1;

    return days;
}

/*
 * Calculates the minutes offset from the 1970 epoch.
 */
NPY_NO_EXPORT npy_int64
get_datetimestruct_minutes(const npy_datetimestruct *dts)
{
    npy_int64 days = get_datetimestruct_days(dts) * 24 * 60;
    days += dts->hour * 60;
    days += dts->min;

    return days;
}

/*
 * Modifies '*days_' to be the day offset within the year,
 * and returns the year.
 */
static npy_int64
days_to_yearsdays(npy_int64 *days_)
{
    const npy_int64 days_per_400years = (400*365 + 100 - 4 + 1);
    /* Adjust so it's relative to the year 2000 (divisible by 400) */
    npy_int64 days = (*days_) - (365*30 + 7);
    npy_int64 year;

    /* Break down the 400 year cycle to get the year and day within the year */
    if (days >= 0) {
        year = 400 * (days / days_per_400years);
        days = days % days_per_400years;
    }
    else {
        year = 400 * ((days - (days_per_400years - 1)) / days_per_400years);
        days = days % days_per_400years;
        if (days < 0) {
            days += days_per_400years;
        }
    }

    /* Work out the year/day within the 400 year cycle */
    if (days >= 366) {
        year += 100 * ((days-1) / (100*365 + 25 - 1));
        days = (days-1) % (100*365 + 25 - 1);
        if (days >= 365) {
            year += 4 * ((days+1) / (4*365 + 1));
            days = (days+1) % (4*365 + 1);
            if (days >= 366) {
                year += (days-1) / 365;
                days = (days-1) % 365;
            }
        }
    }

#ifdef DATETIME_DEBUG
    fprintf(stderr, "days_to_yearsdays: %lld -> %lld, return %lld\n", (long long)*days_, (long long)days, (long long)(year + 2000));
#endif
    *days_ = days;
    return year + 2000;
}

/* Extracts the month number from a 'datetime64[D]' value */
NPY_NO_EXPORT int
days_to_month_number(npy_datetime days)
{
    npy_int64 year;
    int *month_lengths, i;

    year = days_to_yearsdays(&days);
    month_lengths = _days_per_month_table[is_leapyear(year)];

    for (i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            return i + 1;
        }
        else {
            days -= month_lengths[i];
        }
    }

    /* Should never get here */
    return 1;
}

/*
 * Fills in the year, month, day in 'dts' based on the days
 * offset from 1970.
 */
static void
set_datetimestruct_days(npy_int64 days, npy_datetimestruct *dts)
{
    int *month_lengths, i;

    dts->year = days_to_yearsdays(&days);
    month_lengths = _days_per_month_table[is_leapyear(dts->year)];

    for (i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            dts->month = i + 1;
            dts->day = days + 1;
#ifdef DATETIME_DEBUG
            fprintf(stderr, "set_datetimestruct_days: setting month = %d, day = %lld\n", i+1, (long long)(days + 1));
#endif
            return;
        }
        else {
            days -= month_lengths[i];
        }
    }
}

/* get the number of weekdays between 'first' and 'second', where they represent
 * the number of days since 1970-01-01 */
static int
get_nweekdays(npy_int64 first, npy_int64 second)
{
    int dotw_first, dotw_second;
    int ndays;
    int swapped = 0;
    if (second < first) {
        npy_int64 tmp = first;
        first = second;
        second = tmp;
        swapped = 1;
    }
    dotw_first = get_day_of_week(first);
    dotw_second = get_day_of_week(second);
#ifdef DATETIME_DEBUG
    fprintf(stderr, "get_nweekdays: %ld - %ld = ", (long)second, (long)first);
#endif
    if (dotw_first > 4) {
        dotw_first = 4;
    }
    if (dotw_second > 4) {
        dotw_second = 4;
    }
    if (dotw_second < dotw_first) {
        dotw_second += 5;
    }
    ndays = ((second - first) / 7) * 5 + (dotw_second - dotw_first);
    if (swapped) {
        ndays = -ndays;
    }
#ifdef DATETIME_DEBUG
    fprintf(stderr, "%d\n", ndays);
#endif
    return ndays;
}

/*
 * Converts a datetime from a datetimestruct to a datetime based
 * on some metadata. The date is assumed to be valid.
 *
 * TODO: If meta->num is really big, there could be overflow
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_datetimestruct_to_datetime(PyArray_DatetimeMetaData *meta,
                                    const npy_datetimestruct *dts,
                                    npy_datetime *out)
{
    npy_datetime ret;
    NPY_DATETIMEUNIT base = meta->base;

    /* If the datetimestruct is NaT, return NaT */
    if (dts->year == NPY_DATETIME_NAT) {
        *out = NPY_DATETIME_NAT;
        return 0;
    }

    /* Cannot instantiate a datetime with generic units */
    if (meta->base == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot create a NumPy datetime other than NaT "
                    "with generic units");
        return -1;
    }

    if (base == NPY_FR_Y) {
        /* Truncate to the year */
        ret = dts->year - 1970;
    }
    else if (base == NPY_FR_M) {
        /* Truncate to the month */
        ret = 12 * (dts->year - 1970) + (dts->month - 1);
    }
    else {
        /* Otherwise calculate the number of days to start */
        npy_int64 days = get_datetimestruct_days(dts);

        switch (base) {
            case NPY_FR_W:
                /* Truncate to weeks */
                if (days >= 0) {
                    ret = days / 7;
                }
                else {
                    ret = (days - 6) / 7;
                }
                break;
            case NPY_FR_B:
                ret = get_nweekdays(0, days);
#ifdef DATETIME_DEBUG
                fprintf(stderr, "convert_datetimestruct_to_datetime: %d\n",
                        (int)ret);
#endif
                break;
            case NPY_FR_D:
                ret = days;
                break;
            case NPY_FR_h:
                ret = days * 24 +
                      dts->hour;
                break;
            case NPY_FR_m:
                ret = (days * 24 +
                      dts->hour) * 60 +
                      dts->min;
                break;
            case NPY_FR_s:
                ret = ((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec;
                break;
            case NPY_FR_ms:
                ret = (((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000 +
                      dts->us / 1000;
                break;
            case NPY_FR_us:
                ret = (((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us;
                break;
            case NPY_FR_ns:
                ret = ((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000 +
                      dts->ps / 1000;
                break;
            case NPY_FR_ps:
                ret = ((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps;
                break;
            case NPY_FR_fs:
                /* only 2.6 hours */
                ret = (((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps) * 1000 +
                      dts->as / 1000;
                break;
            case NPY_FR_as:
                /* only 9.2 secs */
                ret = (((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps) * 1000000 +
                      dts->as;
                break;
            default:
                /* Something got corrupted */
                PyErr_SetString(PyExc_ValueError,
                        "NumPy datetime metadata with corrupt unit value");
                return -1;
        }
    }

    /* Divide by the multiplier */
    if (meta->num > 1) {
        if (ret >= 0) {
            ret /= meta->num;
        }
        else {
            ret = (ret - meta->num + 1) / meta->num;
        }
    }

    *out = ret;

    return 0;
}

/*NUMPY_API
 * Create a datetime value from a filled datetime struct and resolution unit.
 *
 * TO BE REMOVED - NOT USED INTERNALLY.
 */
NPY_NO_EXPORT npy_datetime
PyArray_DatetimeStructToDatetime(NPY_DATETIMEUNIT fr, npy_datetimestruct *d)
{
    PyErr_SetString(PyExc_RuntimeError,
            "The NumPy PyArray_DatetimeStructToDatetime function has "
            "been removed");
    return -1;
}

/*NUMPY_API
 * Create a timdelta value from a filled timedelta struct and resolution unit.
 *
 * TO BE REMOVED - NOT USED INTERNALLY.
 */
NPY_NO_EXPORT npy_datetime
PyArray_TimedeltaStructToTimedelta(NPY_DATETIMEUNIT fr, npy_timedeltastruct *d)
{
    PyErr_SetString(PyExc_RuntimeError,
            "The NumPy PyArray_TimedeltaStructToTimedelta function has "
            "been removed");
    return -1;
}

/*
 * Converts a datetime based on the given metadata into a datetimestruct
 */
NPY_NO_EXPORT int
convert_datetime_to_datetimestruct(PyArray_DatetimeMetaData *meta,
                                    npy_datetime dt,
                                    npy_datetimestruct *out)
{
    npy_int64 absdays;
    npy_int64 perday;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->year = 1970;
    out->month = 1;
    out->day = 1;

    /* NaT is signaled in the year */
    if (dt == NPY_DATETIME_NAT) {
        out->year = NPY_DATETIME_NAT;
        return 0;
    }

    /* Datetimes can't be in generic units */
    if (meta->base == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot convert a NumPy datetime value other than NaT "
                    "with generic units");
        return -1;
    }

    /* TODO: Change to a mechanism that avoids the potential overflow */
    dt *= meta->num;

    /*
     * Note that care must be taken with the / and % operators
     * for negative values.
     */
    switch (meta->base) {
        case NPY_FR_Y:
            out->year = 1970 + dt;
            break;

        case NPY_FR_M:
            if (dt >= 0) {
                out->year  = 1970 + dt / 12;
                out->month = dt % 12 + 1;
            }
            else {
                out->year  = 1969 + (dt + 1) / 12;
                out->month = 12 + (dt + 1)% 12;
            }
            break;

        case NPY_FR_W:
            /* A week is 7 days */
            set_datetimestruct_days(dt * 7, out);
            break;

        case NPY_FR_B:
            /* Number of business days since Thursday, 1-1-70 */
            /*
             * A business day is M T W Th F (i.e. all but Sat and Sun.)
             * Convert the business day to the number of actual days.
             *
             * Must convert [0,1,2,3,4,5,6,7,...] to
             *                  [0,1,4,5,6,7,8,11,...]
             * and  [...,-9,-8,-7,-6,-5,-4,-3,-2,-1,0] to
             *        [...,-13,-10,-9,-8,-7,-6,-3,-2,-1,0]
             */
            if (dt >= 0) {
                absdays = 7 * ((dt + 3) / 5) + ((dt + 3) % 5) - 3;
            }
            else {
                /* Recall how C computes / and % with negative numbers */
                absdays = 7 * ((dt - 1) / 5) + ((dt - 1) % 5) + 1;
            }
#ifdef DATETIME_DEBUG
            fprintf(stderr, "convert_datetime_to_datetimestruct: converted "
                    "dt=%d to %d\n", (int)dt, (int)absdays);
#endif
            set_datetimestruct_days(absdays, out);
            break;

        case NPY_FR_D:
            set_datetimestruct_days(dt, out);
            break;

        case NPY_FR_h:
            perday = 24LL;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt;
            break;

        case NPY_FR_m:
            perday = 24LL * 60;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt / 60;
            out->min = dt % 60;
            break;

        case NPY_FR_s:
            perday = 24LL * 60 * 60;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt / (60*60);
            out->min = (dt / 60) % 60;
            out->sec = dt % 60;
            break;

        case NPY_FR_ms:
            perday = 24LL * 60 * 60 * 1000;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt / (60*60*1000LL);
            out->min = (dt / (60*1000LL)) % 60;
            out->sec = (dt / 1000LL) % 60;
            out->us = (dt % 1000LL) * 1000;
            break;

        case NPY_FR_us:
            perday = 24LL * 60LL * 60LL * 1000LL * 1000LL;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt / (60*60*1000000LL);
            out->min = (dt / (60*1000000LL)) % 60;
            out->sec = (dt / 1000000LL) % 60;
            out->us = dt % 1000000LL;
            break;

        case NPY_FR_ns:
            perday = 24LL * 60LL * 60LL * 1000LL * 1000LL * 1000LL;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt / (60*60*1000000000LL);
            out->min = (dt / (60*1000000000LL)) % 60;
            out->sec = (dt / 1000000000LL) % 60;
            out->us = (dt / 1000LL) % 1000000LL;
            out->ps = (dt % 1000LL) * 1000;
            break;

        case NPY_FR_ps:
            perday = 24LL * 60 * 60 * 1000 * 1000 * 1000 * 1000;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt / (60*60*1000000000000LL);
            out->min = (dt / (60*1000000000000LL)) % 60;
            out->sec = (dt / 1000000000000LL) % 60;
            out->us = (dt / 1000000LL) % 1000000LL;
            out->ps = dt % 1000000LL;
            break;

        case NPY_FR_fs:
            /* entire range is only +- 2.6 hours */
            if (dt >= 0) {
                out->hour = dt / (60*60*1000000000000000LL);
                out->min = (dt / (60*1000000000000000LL)) % 60;
                out->sec = (dt / 1000000000000000LL) % 60;
                out->us = (dt / 1000000000LL) % 1000000LL;
                out->ps = (dt / 1000LL) % 1000000LL;
                out->as = (dt % 1000LL) * 1000;
            }
            else {
                npy_datetime minutes;

                minutes = dt / (60*1000000000000000LL);
                dt = dt % (60*1000000000000000LL);
                if (dt < 0) {
                    dt += (60*1000000000000000LL);
                    --minutes;
                }
                /* Offset the negative minutes */
                add_minutes_to_datetimestruct(out, minutes);
                out->sec = (dt / 1000000000000000LL) % 60;
                out->us = (dt / 1000000000LL) % 1000000LL;
                out->ps = (dt / 1000LL) % 1000000LL;
                out->as = (dt % 1000LL) * 1000;
            }
            break;

        case NPY_FR_as:
            /* entire range is only +- 9.2 seconds */
            if (dt >= 0) {
                out->sec = (dt / 1000000000000000000LL) % 60;
                out->us = (dt / 1000000000000LL) % 1000000LL;
                out->ps = (dt / 1000000LL) % 1000000LL;
                out->as = dt % 1000000LL;
            }
            else {
                npy_datetime seconds;

                seconds = dt / 1000000000000000000LL;
                dt = dt % 1000000000000000000LL;
                if (dt < 0) {
                    dt += 1000000000000000000LL;
                    --seconds;
                }
                /* Offset the negative seconds */
                add_seconds_to_datetimestruct(out, seconds);
                out->us = (dt / 1000000000000LL) % 1000000LL;
                out->ps = (dt / 1000000LL) % 1000000LL;
                out->as = dt % 1000000LL;
            }
            break;

        default:
            PyErr_SetString(PyExc_RuntimeError,
                        "NumPy datetime metadata is corrupted with invalid "
                        "base unit");
            return -1;
    }

    return 0;
}


/*NUMPY_API
 * Fill the datetime struct from the value and resolution unit.
 *
 * TO BE REMOVED - NOT USED INTERNALLY.
 */
NPY_NO_EXPORT void
PyArray_DatetimeToDatetimeStruct(npy_datetime val, NPY_DATETIMEUNIT fr,
                                 npy_datetimestruct *result)
{
    PyErr_SetString(PyExc_RuntimeError,
            "The NumPy PyArray_DatetimeToDatetimeStruct function has "
            "been removed");
    memset(result, -1, sizeof(npy_datetimestruct));
}

/*
 * FIXME: Overflow is not handled at all
 *   To convert from Years, Months, and Business Days,
 *   multiplication by the average is done
 */

/*NUMPY_API
 * Fill the timedelta struct from the timedelta value and resolution unit.
 *
 * TO BE REMOVED - NOT USED INTERNALLY.
 */
NPY_NO_EXPORT void
PyArray_TimedeltaToTimedeltaStruct(npy_timedelta val, NPY_DATETIMEUNIT fr,
                                 npy_timedeltastruct *result)
{
    PyErr_SetString(PyExc_RuntimeError,
            "The NumPy PyArray_TimedeltaToTimedeltaStruct function has "
            "been removed");
    memset(result, -1, sizeof(npy_timedeltastruct));
}

/*
 * Creates a datetime or timedelta dtype using a copy of the provided metadata.
 */
NPY_NO_EXPORT PyArray_Descr *
create_datetime_dtype(int type_num, PyArray_DatetimeMetaData *meta)
{
    PyArray_Descr *dtype = NULL;
    PyArray_DatetimeMetaData *dt_data;
    PyObject *metacobj = NULL;

    /* Create a default datetime or timedelta */
    if (type_num == NPY_DATETIME || type_num == NPY_TIMEDELTA) {
        dtype = PyArray_DescrNewFromType(type_num);
    }
    else {
        PyErr_SetString(PyExc_RuntimeError,
                "Asked to create a datetime type with a non-datetime "
                "type number");
        return NULL;
    }

    if (dtype == NULL) {
        return NULL;
    }

    /*
     * Remove any reference to old metadata dictionary
     * And create a new one for this new dtype
     */
    Py_XDECREF(dtype->metadata);
    dtype->metadata = PyDict_New();
    if (dtype->metadata == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }

    /* Create a metadata capsule to copy the provided metadata */
    dt_data = PyArray_malloc(sizeof(PyArray_DatetimeMetaData));
    if (dt_data == NULL) {
        Py_DECREF(dtype);
        PyErr_NoMemory();
        return NULL;
    }

    /* Copy the metadata */
    *dt_data = *meta;

    /* Allocate a capsule for it (this claims ownership of dt_data) */
    metacobj = NpyCapsule_FromVoidPtr((void *)dt_data, simple_capsule_dtor);
    if (metacobj == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }

    /* Set the metadata object in the dictionary. */
    if (PyDict_SetItemString(dtype->metadata, NPY_METADATA_DTSTR,
                                                    metacobj) < 0) {
        Py_DECREF(dtype);
        Py_DECREF(metacobj);
        return NULL;
    }
    Py_DECREF(metacobj);

    return dtype;
}

/*
 * Creates a datetime or timedelta dtype using the given unit.
 */
NPY_NO_EXPORT PyArray_Descr *
create_datetime_dtype_with_unit(int type_num, NPY_DATETIMEUNIT unit)
{
    PyArray_DatetimeMetaData meta;
    meta.base = unit;
    meta.num = 1;
    return create_datetime_dtype(type_num, &meta);
}

/*
 * This function returns the a new reference to the
 * capsule with the datetime metadata.
 */
NPY_NO_EXPORT PyObject *
get_datetime_metacobj_from_dtype(PyArray_Descr *dtype)
{
    PyObject *metacobj;

    /* Check that the dtype has metadata */
    if (dtype->metadata == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Datetime type object is invalid, lacks metadata");
        return NULL;
    }

    /* Check that the dtype has unit metadata */
    metacobj = PyDict_GetItemString(dtype->metadata, NPY_METADATA_DTSTR);
    if (metacobj == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Datetime type object is invalid, lacks unit metadata");
        return NULL;
    }

    Py_INCREF(metacobj);
    return metacobj;
}

/*
 * This function returns a pointer to the DateTimeMetaData
 * contained within the provided datetime dtype.
 */
NPY_NO_EXPORT PyArray_DatetimeMetaData *
get_datetime_metadata_from_dtype(PyArray_Descr *dtype)
{
    PyObject *metacobj;
    PyArray_DatetimeMetaData *meta = NULL;

    metacobj = get_datetime_metacobj_from_dtype(dtype);
    if (metacobj == NULL) {
        return NULL;
    }

    /* Check that the dtype has an NpyCapsule for the metadata */
    meta = (PyArray_DatetimeMetaData *)NpyCapsule_AsVoidPtr(metacobj);
    if (meta == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Datetime type object is invalid, unit metadata is corrupt");
        return NULL;
    }

    return meta;
}

/*
 * Converts a substring given by 'str' and 'len' into
 * a date time unit multiplier + enum value, which are populated
 * into out_meta. Other metadata is left along.
 *
 * 'metastr' is only used in the error message, and may be NULL.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
parse_datetime_extended_unit_from_string(char *str, Py_ssize_t len,
                                    char *metastr,
                                    PyArray_DatetimeMetaData *out_meta)
{
    char *substr = str, *substrend = NULL;
    int den = 1;

    /* First comes an optional integer multiplier */
    out_meta->num = (int)strtol(substr, &substrend, 10);
    if (substr == substrend) {
        out_meta->num = 1;
    }
    substr = substrend;

    /* Next comes the unit itself, followed by either '/' or the string end */
    substrend = substr;
    while (substrend-str < len && *substrend != '/') {
        ++substrend;
    }
    if (substr == substrend) {
        goto bad_input;
    }
    out_meta->base = parse_datetime_unit_from_string(substr,
                                        substrend-substr, metastr);
    if (out_meta->base == -1) {
        return -1;
    }
    substr = substrend;

    /* Next comes an optional integer denominator */
    if (substr-str < len && *substr == '/') {
        substr++;
        den = (int)strtol(substr, &substrend, 10);
        /* If the '/' exists, there must be a number followed by ']' */
        if (substr == substrend || *substrend != ']') {
            goto bad_input;
        }
        substr = substrend + 1;
    }
    else if (substr-str != len) {
        goto bad_input;
    }

    if (den != 1) {
        if (convert_datetime_divisor_to_multiple(
                                out_meta, den, metastr) < 0) {
            return -1;
        }
    }

    return 0;

bad_input:
    if (metastr != NULL) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\" at position %d",
                metastr, (int)(substr-metastr));
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\"",
                str);
    }

    return -1;
}

/*
 * Parses the metadata string into the metadata C structure.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
parse_datetime_metadata_from_metastr(char *metastr, Py_ssize_t len,
                                    PyArray_DatetimeMetaData *out_meta)
{
    char *substr = metastr, *substrend = NULL;

    /* Treat the empty string as generic units */
    if (len == 0) {
        out_meta->base = NPY_FR_GENERIC;
        out_meta->num = 1;

        return 0;
    }

    /* The metadata string must start with a '[' */
    if (len < 3 || *substr++ != '[') {
        goto bad_input;
    }

    substrend = substr;
    while (substrend - metastr < len && *substrend != ']') {
        ++substrend;
    }
    if (substrend - metastr == len || substr == substrend) {
        substr = substrend;
        goto bad_input;
    }

    /* Parse the extended unit inside the [] */
    if (parse_datetime_extended_unit_from_string(substr, substrend-substr,
                                                    metastr, out_meta) < 0) {
        return -1;
    }

    substr = substrend+1;

    if (substr - metastr != len) {
        goto bad_input;
    }

    return 0;

bad_input:
    if (substr != metastr) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\" at position %d",
                metastr, (int)(substr-metastr));
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\"",
                metastr);
    }

    return -1;
}

/*
 * Converts a datetype dtype string into a dtype descr object.
 * The "type" string should be NULL-terminated.
 */
NPY_NO_EXPORT PyArray_Descr *
parse_dtype_from_datetime_typestr(char *typestr, Py_ssize_t len)
{
    PyArray_DatetimeMetaData meta;
    char *metastr = NULL;
    int is_timedelta = 0;
    Py_ssize_t metalen = 0;

    if (len < 2) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime typestr \"%s\"",
                typestr);
        return NULL;
    }

    /*
     * First validate that the root is correct,
     * and get the metadata string address
     */
    if (typestr[0] == 'm' && typestr[1] == '8') {
        is_timedelta = 1;
        metastr = typestr + 2;
        metalen = len - 2;
    }
    else if (typestr[0] == 'M' && typestr[1] == '8') {
        is_timedelta = 0;
        metastr = typestr + 2;
        metalen = len - 2;
    }
    else if (len >= 11 && strncmp(typestr, "timedelta64", 11) == 0) {
        is_timedelta = 1;
        metastr = typestr + 11;
        metalen = len - 11;
    }
    else if (len >= 10 && strncmp(typestr, "datetime64", 10) == 0) {
        is_timedelta = 0;
        metastr = typestr + 10;
        metalen = len - 10;
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime typestr \"%s\"",
                typestr);
        return NULL;
    }

    /* Parse the metadata string into a metadata struct */
    if (parse_datetime_metadata_from_metastr(metastr, metalen, &meta) < 0) {
        return NULL;
    }

    return create_datetime_dtype(is_timedelta ? NPY_TIMEDELTA : NPY_DATETIME,
                                    &meta);
}

static NPY_DATETIMEUNIT _multiples_table[16][4] = {
    {12, 52, 365},                            /* NPY_FR_Y */
    {NPY_FR_M, NPY_FR_W, NPY_FR_D},
    {4,  30, 720},                            /* NPY_FR_M */
    {NPY_FR_W, NPY_FR_D, NPY_FR_h},
    {5, 7,  168, 10080},                      /* NPY_FR_W */
    {NPY_FR_B, NPY_FR_D, NPY_FR_h, NPY_FR_m},
    {24, 1440, 86400},                        /* NPY_FR_B */
    {NPY_FR_h, NPY_FR_m, NPY_FR_s},
    {24, 1440, 86400},                        /* NPY_FR_D */
    {NPY_FR_h, NPY_FR_m, NPY_FR_s},
    {60, 3600},                               /* NPY_FR_h */
    {NPY_FR_m, NPY_FR_s},
    {60, 60000},                              /* NPY_FR_m */
    {NPY_FR_s, NPY_FR_ms},
    {1000, 1000000},                          /* >=NPY_FR_s */
    {0, 0}
};



/*
 * Translate divisors into multiples of smaller units.
 * 'metastr' is used for the error message if the divisor doesn't work,
 * and can be NULL if the metadata didn't come from a string.
 *
 * This function only affects the 'base' and 'num' values in the metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_datetime_divisor_to_multiple(PyArray_DatetimeMetaData *meta,
                                    int den, char *metastr)
{
    int i, num, ind;
    NPY_DATETIMEUNIT *totry;
    NPY_DATETIMEUNIT *baseunit;
    int q, r;

    if (meta->base == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
            "Can't use 'den' divisor with generic units");
        return -1;
    }

    ind = ((int)meta->base - (int)NPY_FR_Y)*2;
    totry = _multiples_table[ind];
    baseunit = _multiples_table[ind + 1];

    num = 3;
    if (meta->base == NPY_FR_W) {
        num = 4;
    }
    else if (meta->base > NPY_FR_D) {
        num = 2;
    }
    if (meta->base >= NPY_FR_s) {
        ind = ((int)NPY_FR_s - (int)NPY_FR_Y)*2;
        totry = _multiples_table[ind];
        baseunit = _multiples_table[ind + 1];
        baseunit[0] = meta->base + 1;
        baseunit[1] = meta->base + 2;
        if (meta->base == NPY_FR_as - 1) {
            num = 1;
        }
        if (meta->base == NPY_FR_as) {
            num = 0;
        }
    }

    for (i = 0; i < num; i++) {
        q = totry[i] / den;
        r = totry[i] % den;
        if (r == 0) {
            break;
        }
    }
    if (i == num) {
        if (metastr == NULL) {
            PyErr_Format(PyExc_ValueError,
                    "divisor (%d) is not a multiple of a lower-unit "
                    "in datetime metadata", den);
        }
        else {
            PyErr_Format(PyExc_ValueError,
                    "divisor (%d) is not a multiple of a lower-unit "
                    "in datetime metadata \"%s\"", den, metastr);
        }
        return -1;
    }
    meta->base = baseunit[i];
    meta->num *= q;

    return 0;
}

/*
 * Lookup table for factors between datetime units, except
 * for years, months and business days.
 */
static npy_uint32
_datetime_factors[] = {
    1,  /* Years - not used */
    1,  /* Months - not used */
    7,  /* Weeks -> Days */
    1,  /* Business days - not used */
    24, /* Days -> Hours */
    60, /* Hours -> Minutes */
    60, /* Minutes -> Seconds */
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1,   /* Attoseconds are the smallest base unit */
    0    /* Generic units don't have a conversion */
};

/*
 * Returns the scale factor between the units. Does not validate
 * that bigbase represents larger units than littlebase, or that
 * the units are not generic.
 *
 * Returns 0 if there is an overflow.
 */
static npy_uint64
get_datetime_units_factor(NPY_DATETIMEUNIT bigbase, NPY_DATETIMEUNIT littlebase)
{
    npy_uint64 factor = 1;
    int unit = (int)bigbase;
    while (littlebase > unit) {
        factor *= _datetime_factors[unit];
        /*
         * Detect overflow by disallowing the top 16 bits to be 1.
         * That alows a margin of error much bigger than any of
         * the datetime factors.
         */
        if (factor&0xff00000000000000ULL) {
            return 0;
        }
        ++unit;
    }
    return factor;
}

/* Euclidean algorithm on two positive numbers */
static npy_uint64
_uint64_euclidean_gcd(npy_uint64 x, npy_uint64 y)
{
    npy_uint64 tmp;

    if (x > y) {
        tmp = x;
        x = y;
        y = tmp;
    }
    while (x != y && y != 0) {
        tmp = x % y;
        x = y;
        y = tmp;
    }

    return x;
}

/*
 * Computes the conversion factor to convert data with 'src_meta' metadata
 * into data with 'dst_meta' metadata.
 *
 * If overflow occurs, both out_num and out_denom are set to 0, but
 * no error is set.
 */
NPY_NO_EXPORT void
get_datetime_conversion_factor(PyArray_DatetimeMetaData *src_meta,
                                PyArray_DatetimeMetaData *dst_meta,
                                npy_int64 *out_num, npy_int64 *out_denom)
{
    int src_base, dst_base, swapped;
    npy_uint64 num = 1, denom = 1, tmp, gcd;

    /* Generic units change to the destination with no conversion factor */
    if (src_meta->base == NPY_FR_GENERIC) {
        *out_num = 1;
        *out_denom = 1;
        return;
    }
    /*
     * Converting to a generic unit from something other than a generic
     * unit is an error.
     */
    else if (dst_meta->base == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot convert from specific units to generic "
                    "units in NumPy datetimes or timedeltas");
        *out_num = 0;
        *out_denom = 0;
        return;
    }

    if (src_meta->base <= dst_meta->base) {
        src_base = src_meta->base;
        dst_base = dst_meta->base;
        swapped = 0;
    }
    else {
        src_base = dst_meta->base;
        dst_base = src_meta->base;
        swapped = 1;
    }

    if (src_base != dst_base) {
        /*
         * Conversions between years/months and other units use
         * the factor averaged over the 400 year leap year cycle.
         */
        if (src_base == NPY_FR_Y) {
            if (dst_base == NPY_FR_M) {
                num *= 12;
            }
            else if (dst_base == NPY_FR_W) {
                num *= (97 + 400*365);
                denom *= 400*7;
            }
            else if (dst_base == NPY_FR_B) {
                /* Luckily, 97 + 400*365 is divisible by 7, we can calculate
                 * the number of business days in 400 years exactly. */
                num *= (97 + 400*365) * 5 / 7;
                denom *= 400;
                /* Business Day -> dst_base */
                num *= get_datetime_units_factor(NPY_FR_B, dst_base);
            }
            else {
                /* Year -> Day */
                num *= (97 + 400*365);
                denom *= 400;
                /* Day -> dst_base */
                num *= get_datetime_units_factor(NPY_FR_D, dst_base);
            }
        }
        else if (src_base == NPY_FR_M) {
            if (dst_base == NPY_FR_W) {
                num *= (97 + 400*365);
                denom *= 400*12*7;
            }
            else {
                /* Month -> Day */
                num *= (97 + 400*365);
                denom *= 400*12;
                if (dst_base == NPY_FR_B) {
                    num *= 5;
                    denom *= 7;
                }
                /* Day -> dst_base */
                num *= get_datetime_units_factor(NPY_FR_D, dst_base);
            }
        }
        else {
            num *= get_datetime_units_factor(src_base, dst_base);
        }
    }

    /* If something overflowed, make both num and denom 0 */
    if (denom == 0) {
        PyErr_Format(PyExc_OverflowError,
                    "Integer overflow while computing the conversion "
                    "factor between NumPy datetime units %s and %s",
                    _datetime_strings[src_base],
                    _datetime_strings[dst_base]);
        *out_num = 0;
        *out_denom = 0;
        return;
    }

    /* Swap the numerator and denominator if necessary */
    if (swapped) {
        tmp = num;
        num = denom;
        denom = tmp;
    }

    num *= src_meta->num;
    denom *= dst_meta->num;

    /* Return as a fraction in reduced form */
    gcd = _uint64_euclidean_gcd(num, denom);
    *out_num = (npy_int64)(num / gcd);
    *out_denom = (npy_int64)(denom / gcd);
}

/*
 * Determines whether the 'divisor' metadata divides evenly into
 * the 'dividend' metadata.
 */
NPY_NO_EXPORT npy_bool
datetime_metadata_divides(
                        PyArray_DatetimeMetaData *dividend,
                        PyArray_DatetimeMetaData *divisor,
                        int strict_with_nonlinear_units)
{
    npy_uint64 num1, num2;

    /* Generic units divide into anything */
    if (divisor->base == NPY_FR_GENERIC) {
        return 1;
    }
    /* Non-generic units never divide into generic units */
    else if (dividend->base == NPY_FR_GENERIC) {
        return 0;
    }

    num1 = (npy_uint64)dividend->num;
    num2 = (npy_uint64)divisor->num;

    /* If the bases are different, factor in a conversion */
    if (dividend->base != divisor->base) {
        /*
         * Years, Months and Business days are incompatible with
         * all other units (except years and months are compatible
         * with each other).
         */
        if (dividend->base == NPY_FR_B || divisor->base == NPY_FR_B) {
#ifdef DATETIME_DEBUG
            fprintf(stderr, "datetime_metadata_divides: 0\n");
#endif
            return 0;
        }
        else if (dividend->base == NPY_FR_Y) {
            if (divisor->base == NPY_FR_M) {
                num1 *= 12;
            }
            else if (strict_with_nonlinear_units) {
                return 0;
            }
            else {
                /* Could do something complicated here */
                return 1;
            }
        }
        else if (divisor->base == NPY_FR_Y) {
            if (dividend->base == NPY_FR_M) {
                num2 *= 12;
            }
            else if (strict_with_nonlinear_units) {
                return 0;
            }
            else {
                /* Could do something complicated here */
                return 1;
            }
        }
        else if (dividend->base == NPY_FR_M || divisor->base == NPY_FR_M) {
            if (strict_with_nonlinear_units) {
                return 0;
            }
            else {
                /* Could do something complicated here */
                return 1;
            }
        }

        /* Take the greater base (unit sizes are decreasing in enum) */
        if (dividend->base > divisor->base) {
            num2 *= get_datetime_units_factor(divisor->base, dividend->base);
            if (num2 == 0) {
                return 0;
            }
        }
        else {
            num1 *= get_datetime_units_factor(dividend->base, divisor->base);
            if (num1 == 0) {
                return 0;
            }
        }
    }

    /* Crude, incomplete check for overflow */
    if (num1&0xff00000000000000LL || num2&0xff00000000000000LL ) {
        return 0;
    }

    return (num1 % num2) == 0;
}

/*
 * This provides the casting rules for the DATETIME data type units.
 *
 * Notably, there is a barrier between 'date units' and 'time units'
 * for all but 'unsafe' casting.
 */
NPY_NO_EXPORT npy_bool
can_cast_datetime64_units(NPY_DATETIMEUNIT src_unit,
                          NPY_DATETIMEUNIT dst_unit,
                          NPY_CASTING casting)
{
    switch (casting) {
        /* Allow anything with unsafe casting */
        case NPY_UNSAFE_CASTING:
            return 1;

        /*
         * Only enforce the 'date units' vs 'time units' barrier with
         * 'same_kind' casting.
         */
        case NPY_SAME_KIND_CASTING:
            if (src_unit == NPY_FR_GENERIC || dst_unit == NPY_FR_GENERIC) {
                return src_unit == dst_unit;
            }
            else {
                return (src_unit <= NPY_FR_D && dst_unit <= NPY_FR_D) ||
                       (src_unit > NPY_FR_D && dst_unit > NPY_FR_D);
            }

        /*
         * Enforce the 'date units' vs 'time units' barrier and that
         * casting is only allowed towards more precise units with
         * 'safe' casting.
         */
        case NPY_SAFE_CASTING:
            if (src_unit == NPY_FR_GENERIC || dst_unit == NPY_FR_GENERIC) {
                return src_unit == dst_unit;
            }
            else {
                return (src_unit <= dst_unit) &&
                       ((src_unit <= NPY_FR_D && dst_unit <= NPY_FR_D) ||
                        (src_unit > NPY_FR_D && dst_unit > NPY_FR_D));
            }

        /* Enforce equality with 'no' or 'equiv' casting */
        default:
            return src_unit == dst_unit;
    }
}

/*
 * This provides the casting rules for the TIMEDELTA data type units.
 *
 * Notably, there is a barrier between the nonlinear years and
 * months units, and all the other units.
 */
NPY_NO_EXPORT npy_bool
can_cast_timedelta64_units(NPY_DATETIMEUNIT src_unit,
                          NPY_DATETIMEUNIT dst_unit,
                          NPY_CASTING casting)
{
    switch (casting) {
        /* Allow anything with unsafe casting */
        case NPY_UNSAFE_CASTING:
            return 1;

        /*
         * Only enforce the 'date units' vs 'time units' barrier with
         * 'same_kind' casting.
         */
        case NPY_SAME_KIND_CASTING:
            if (src_unit == NPY_FR_GENERIC || dst_unit == NPY_FR_GENERIC) {
                return src_unit == dst_unit;
            }
            else {
                return (src_unit <= NPY_FR_M && dst_unit <= NPY_FR_M) ||
                       (src_unit > NPY_FR_M && dst_unit > NPY_FR_M);
            }

        /*
         * Enforce the 'date units' vs 'time units' barrier and that
         * casting is only allowed towards more precise units with
         * 'safe' casting.
         */
        case NPY_SAFE_CASTING:
            if (src_unit == NPY_FR_GENERIC || dst_unit == NPY_FR_GENERIC) {
                return src_unit == dst_unit;
            }
            else {
                return (src_unit <= dst_unit) &&
                       ((src_unit <= NPY_FR_M && dst_unit <= NPY_FR_M) ||
                        (src_unit > NPY_FR_M && dst_unit > NPY_FR_M));
            }

        /* Enforce equality with 'no' or 'equiv' casting */
        default:
            return src_unit == dst_unit;
    }
}

/*
 * This provides the casting rules for the DATETIME data type metadata.
 */
NPY_NO_EXPORT npy_bool
can_cast_datetime64_metadata(PyArray_DatetimeMetaData *src_meta,
                             PyArray_DatetimeMetaData *dst_meta,
                             NPY_CASTING casting)
{
    switch (casting) {
        case NPY_UNSAFE_CASTING:
            return 1;

        case NPY_SAME_KIND_CASTING:
            return can_cast_datetime64_units(src_meta->base, dst_meta->base,
                                             casting);

        case NPY_SAFE_CASTING:
            return can_cast_datetime64_units(src_meta->base, dst_meta->base,
                                                             casting) &&
                   datetime_metadata_divides(src_meta, dst_meta, 0);

        default:
            return src_meta->base == dst_meta->base &&
                   src_meta->num == dst_meta->num;
    }
}

/*
 * This provides the casting rules for the TIMEDELTA data type metadata.
 */
NPY_NO_EXPORT npy_bool
can_cast_timedelta64_metadata(PyArray_DatetimeMetaData *src_meta,
                             PyArray_DatetimeMetaData *dst_meta,
                             NPY_CASTING casting)
{
    switch (casting) {
        case NPY_UNSAFE_CASTING:
            return 1;

        case NPY_SAME_KIND_CASTING:
            return can_cast_timedelta64_units(src_meta->base, dst_meta->base,
                                             casting);

        case NPY_SAFE_CASTING:
            return can_cast_timedelta64_units(src_meta->base, dst_meta->base,
                                                             casting) &&
                   datetime_metadata_divides(src_meta, dst_meta, 1);

        default:
            return src_meta->base == dst_meta->base &&
                   src_meta->num == dst_meta->num;
    }
}

/*
 * Tests whether a datetime64 can be cast from the source metadata
 * to the destination metadata according to the specified casting rule.
 *
 * Returns -1 if an exception was raised, 0 otherwise.
 */
NPY_NO_EXPORT int
raise_if_datetime64_metadata_cast_error(char *object_type,
                            PyArray_DatetimeMetaData *src_meta,
                            PyArray_DatetimeMetaData *dst_meta,
                            NPY_CASTING casting)
{
    if (can_cast_datetime64_metadata(src_meta, dst_meta, casting)) {
        return 0;
    }
    else {
        PyObject *errmsg;
        errmsg = PyUString_FromFormat("Cannot cast %s "
                    "from metadata ", object_type);
        errmsg = append_metastr_to_string(src_meta, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" to "));
        errmsg = append_metastr_to_string(dst_meta, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromFormat(" according to the rule %s",
                        npy_casting_to_string(casting)));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        return -1;
    }
}

/*
 * Tests whether a timedelta64 can be cast from the source metadata
 * to the destination metadata according to the specified casting rule.
 *
 * Returns -1 if an exception was raised, 0 otherwise.
 */
NPY_NO_EXPORT int
raise_if_timedelta64_metadata_cast_error(char *object_type,
                            PyArray_DatetimeMetaData *src_meta,
                            PyArray_DatetimeMetaData *dst_meta,
                            NPY_CASTING casting)
{
    if (can_cast_timedelta64_metadata(src_meta, dst_meta, casting)) {
        return 0;
    }
    else {
        PyObject *errmsg;
        errmsg = PyUString_FromFormat("Cannot cast %s "
                    "from metadata ", object_type);
        errmsg = append_metastr_to_string(src_meta, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" to "));
        errmsg = append_metastr_to_string(dst_meta, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromFormat(" according to the rule %s",
                        npy_casting_to_string(casting)));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        return -1;
    }
}

/*
 * Computes the GCD of the two date-time metadata values. Raises
 * an exception if there is no reasonable GCD, such as with
 * years and days.
 *
 * The result is placed in 'out_meta'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
compute_datetime_metadata_greatest_common_divisor(
                        PyArray_DatetimeMetaData *meta1,
                        PyArray_DatetimeMetaData *meta2,
                        PyArray_DatetimeMetaData *out_meta,
                        int strict_with_nonlinear_units1,
                        int strict_with_nonlinear_units2)
{
    NPY_DATETIMEUNIT base;
    npy_uint64 num1, num2, num;

    /* If either unit is generic, adopt the metadata from the other one */
    if (meta1->base == NPY_FR_GENERIC) {
        *out_meta = *meta2;
        return 0;
    }
    else if (meta2->base == NPY_FR_GENERIC) {
        *out_meta = *meta1;
        return 0;
    }

    num1 = (npy_uint64)meta1->num;
    num2 = (npy_uint64)meta2->num;

    /* First validate that the units have a reasonable GCD */
    if (meta1->base == meta2->base) {
        base = meta1->base;
    }
    else {
        /*
         * Years, Months and Business days are incompatible with
         * all other units (except years and months are compatible
         * with each other).
         */
        if (meta1->base == NPY_FR_Y) {
            if (meta2->base == NPY_FR_M) {
                base = NPY_FR_M;
                num1 *= 12;
            }
            else if (strict_with_nonlinear_units1) {
                goto incompatible_units;
            }
            else {
                base = meta2->base;
                /* Don't multiply num1 since there is no even factor */
            }
        }
        else if (meta1->base == NPY_FR_B || meta2->base == NPY_FR_B) {
            if (strict_with_nonlinear_units1 || strict_with_nonlinear_units2) {
                goto incompatible_units;
            }
            else {
                if (meta1->base > meta2->base) {
                    base = meta1->base;
                }
                else {
                    base = meta2->base;
                }
#ifdef DATETIME_DEBUG
                fprintf(stderr, "compute_datetime_metadata_greatest_common_divisor: bases: %d, %d, using %d\n", (int)meta1->base, (int)meta2->base, (int)base);
#endif
                /*
                 * When combining business days with other units, end
                 * up with days instead of business days.
                 */
                if (base == NPY_FR_B) {
#ifdef DATETIME_DEBUG
                    fprintf(stderr, "compute_datetime_metadata_greatest_common_divisor: converting business days to regular days\n");
#endif
                    base = NPY_FR_D;
                }
            }
        }
        else if (meta2->base == NPY_FR_Y) {
            if (meta1->base == NPY_FR_M) {
                base = NPY_FR_M;
                num2 *= 12;
            }
            else if (strict_with_nonlinear_units2) {
                goto incompatible_units;
            }
            else {
                base = meta1->base;
                /* Don't multiply num2 since there is no even factor */
            }
        }
        else if (meta1->base == NPY_FR_M) {
            if (strict_with_nonlinear_units1) {
                goto incompatible_units;
            }
            else {
                base = meta2->base;
                /* Don't multiply num1 since there is no even factor */
            }
        }
        else if (meta2->base == NPY_FR_M) {
            if (strict_with_nonlinear_units2) {
                goto incompatible_units;
            }
            else {
                base = meta1->base;
                /* Don't multiply num2 since there is no even factor */
            }
        }

        /* Take the greater base (unit sizes are decreasing in enum) */
        if (meta1->base > meta2->base) {
            base = meta1->base;
            num2 *= get_datetime_units_factor(meta2->base, meta1->base);
            if (num2 == 0) {
                goto units_overflow;
            }
        }
        else {
            base = meta2->base;
            num1 *= get_datetime_units_factor(meta1->base, meta2->base);
            if (num1 == 0) {
                goto units_overflow;
            }
        }
    }

    /* Compute the GCD of the resulting multipliers */
    num = _uint64_euclidean_gcd(num1, num2);

    /* Fill the 'out_meta' values */
    out_meta->base = base;
    out_meta->num = (int)num;
    if (out_meta->num <= 0 || num != (npy_uint64)out_meta->num) {
        goto units_overflow;
    }

    return 0;

incompatible_units: {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Cannot get "
                    "a common metadata divisor for "
                    "NumPy datetime metadata ");
        errmsg = append_metastr_to_string(meta1, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" and "));
        errmsg = append_metastr_to_string(meta2, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" because they have "
                    "incompatible nonlinear base time units"));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        return -1;
    }
units_overflow: {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Integer overflow "
                    "getting a common metadata divisor for "
                    "NumPy datetime metadata ");
        errmsg = append_metastr_to_string(meta1, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" and "));
        errmsg = append_metastr_to_string(meta2, 0, errmsg);
        PyErr_SetObject(PyExc_OverflowError, errmsg);
        return -1;
    }
}

/*
 * Computes the GCD of the two date-time metadata values. Raises
 * an exception if there is no reasonable GCD, such as with
 * years and days.
 *
 * Returns a capsule with the GCD metadata.
 */
NPY_NO_EXPORT PyObject *
compute_datetime_metadata_greatest_common_divisor_capsule(
                        PyArray_Descr *type1,
                        PyArray_Descr *type2,
                        int strict_with_nonlinear_units1,
                        int strict_with_nonlinear_units2)
{
    PyArray_DatetimeMetaData *meta1, *meta2, *dt_data;

    if ((type1->type_num != NPY_DATETIME &&
                        type1->type_num != NPY_TIMEDELTA) ||
                    (type2->type_num != NPY_DATETIME &&
                        type2->type_num != NPY_TIMEDELTA)) {
        PyErr_SetString(PyExc_TypeError,
                "Require datetime types for metadata "
                "greatest common divisor operation");
        return NULL;
    }

    meta1 = get_datetime_metadata_from_dtype(type1);
    if (meta1 == NULL) {
        return NULL;
    }
    meta2 = get_datetime_metadata_from_dtype(type2);
    if (meta2 == NULL) {
        return NULL;
    }

    /* Create and return the metadata capsule */
    dt_data = PyArray_malloc(sizeof(PyArray_DatetimeMetaData));
    if (dt_data == NULL) {
        return PyErr_NoMemory();
    }

    if (compute_datetime_metadata_greatest_common_divisor(meta1, meta2,
                            dt_data, strict_with_nonlinear_units1,
                            strict_with_nonlinear_units2) < 0) {
        PyArray_free(dt_data);
        return NULL;
    }

    return NpyCapsule_FromVoidPtr((void *)dt_data, simple_capsule_dtor);
}

/*
 * Both type1 and type2 must be either NPY_DATETIME or NPY_TIMEDELTA.
 * Applies the type promotion rules between the two types, returning
 * the promoted type.
 */
NPY_NO_EXPORT PyArray_Descr *
datetime_type_promotion(PyArray_Descr *type1, PyArray_Descr *type2)
{
    int type_num1, type_num2;
    PyObject *gcdmeta;
    PyArray_Descr *dtype;
    int is_datetime;

    type_num1 = type1->type_num;
    type_num2 = type2->type_num;

    is_datetime = (type_num1 == NPY_DATETIME || type_num2 == NPY_DATETIME);

    /*
     * Get the metadata GCD, being strict about nonlinear units for
     * timedelta and relaxed for datetime.
     */
    gcdmeta = compute_datetime_metadata_greatest_common_divisor_capsule(
                                            type1, type2,
                                            type_num1 == NPY_TIMEDELTA,
                                            type_num2 == NPY_TIMEDELTA);
    if (gcdmeta == NULL) {
        return NULL;
    }

    /* Create a DATETIME or TIMEDELTA dtype */
    dtype = PyArray_DescrNewFromType(is_datetime ? NPY_DATETIME :
                                                   NPY_TIMEDELTA);
    if (dtype == NULL) {
        Py_DECREF(gcdmeta);
        return NULL;
    }

    /* Replace the metadata dictionary */
    Py_XDECREF(dtype->metadata);
    dtype->metadata = PyDict_New();
    if (dtype->metadata == NULL) {
        Py_DECREF(dtype);
        Py_DECREF(gcdmeta);
        return NULL;
    }

    /* Set the metadata object in the dictionary. */
    if (PyDict_SetItemString(dtype->metadata, NPY_METADATA_DTSTR,
                                                gcdmeta) < 0) {
        Py_DECREF(dtype);
        Py_DECREF(gcdmeta);
        return NULL;
    }
    Py_DECREF(gcdmeta);

    return dtype;


    PyErr_SetString(PyExc_RuntimeError,
            "Called datetime_type_promotion on non-datetype type");
    return NULL;
}

/*
 * Converts a substring given by 'str' and 'len' into
 * a date time unit enum value. The 'metastr' parameter
 * is used for error messages, and may be NULL.
 *
 * Generic units have no representation as a string in this form.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT NPY_DATETIMEUNIT
parse_datetime_unit_from_string(char *str, Py_ssize_t len, char *metastr)
{
    /* Use switch statements so the compiler can make it fast */
    if (len == 1) {
        switch (str[0]) {
            case 'Y':
                return NPY_FR_Y;
            case 'M':
                return NPY_FR_M;
            case 'W':
                return NPY_FR_W;
            case 'B':
                return NPY_FR_B;
            case 'D':
                return NPY_FR_D;
            case 'h':
                return NPY_FR_h;
            case 'm':
                return NPY_FR_m;
            case 's':
                return NPY_FR_s;
        }
    }
    /* All the two-letter units are variants of seconds */
    else if (len == 2 && str[1] == 's') {
        switch (str[0]) {
            case 'm':
                return NPY_FR_ms;
            case 'u':
                return NPY_FR_us;
            case 'n':
                return NPY_FR_ns;
            case 'p':
                return NPY_FR_ps;
            case 'f':
                return NPY_FR_fs;
            case 'a':
                return NPY_FR_as;
        }
    }

    /* If nothing matched, it's an error */
    if (metastr == NULL) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime unit \"%s\" in metadata",
                str);
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime unit in metadata string \"%s\"",
                metastr);
    }
    return -1;
}


NPY_NO_EXPORT PyObject *
convert_datetime_metadata_to_tuple(PyArray_DatetimeMetaData *meta)
{
    PyObject *dt_tuple;

    dt_tuple = PyTuple_New(2);
    if (dt_tuple == NULL) {
        return NULL;
    }

    PyTuple_SET_ITEM(dt_tuple, 0,
            PyBytes_FromString(_datetime_strings[meta->base]));
    PyTuple_SET_ITEM(dt_tuple, 1,
            PyInt_FromLong(meta->num));

    return dt_tuple;
}

/*
 * Converts a metadata tuple into a datetime metadata C struct.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_datetime_metadata_tuple_to_datetime_metadata(PyObject *tuple,
                                        PyArray_DatetimeMetaData *out_meta)
{
    char *basestr = NULL;
    Py_ssize_t len = 0, tuple_size;
    int den = 1;

    if (!PyTuple_Check(tuple)) {
        PyObject_Print(tuple, stderr, 0);
        PyErr_SetString(PyExc_TypeError,
                        "Require tuple for tuple to NumPy datetime "
                        "metadata conversion");
        return -1;
    }

    tuple_size = PyTuple_GET_SIZE(tuple);
    if (tuple_size < 2 || tuple_size > 4) {
        PyErr_SetString(PyExc_TypeError,
                        "Require tuple of size 2 to 4 for "
                        "tuple to NumPy datetime metadata conversion");
        return -1;
    }

    if (PyBytes_AsStringAndSize(PyTuple_GET_ITEM(tuple, 0),
                                        &basestr, &len) < 0) {
        return -1;
    }

    out_meta->base = parse_datetime_unit_from_string(basestr, len, NULL);
    if (out_meta->base == -1) {
        return -1;
    }

    /* Convert the values to longs */
    out_meta->num = PyInt_AsLong(PyTuple_GET_ITEM(tuple, 1));
    if (out_meta->num == -1 && PyErr_Occurred()) {
        return -1;
    }

    if (tuple_size == 4) {
        den = PyInt_AsLong(PyTuple_GET_ITEM(tuple, 2));
        if (den == -1 && PyErr_Occurred()) {
            return -1;
        }
    }

    if (out_meta->num <= 0 || den <= 0) {
        PyErr_SetString(PyExc_TypeError,
                        "Invalid tuple values for "
                        "tuple to NumPy datetime metadata conversion");
        return -1;
    }

    if (den != 1) {
        if (convert_datetime_divisor_to_multiple(out_meta, den, NULL) < 0) {
            return -1;
        }
    }

    return 0;
}

/*
 * Converts a metadata tuple into a datetime metadata capsule.
 */
NPY_NO_EXPORT PyObject *
convert_datetime_metadata_tuple_to_metacobj(PyObject *tuple)
{
    PyArray_DatetimeMetaData *dt_data;

    dt_data = PyArray_malloc(sizeof(PyArray_DatetimeMetaData));

    if (convert_datetime_metadata_tuple_to_datetime_metadata(
                                                tuple, dt_data) < 0) {
        PyArray_free(dt_data);
        return NULL;
    }

    return NpyCapsule_FromVoidPtr((void *)dt_data, simple_capsule_dtor);
}

/*
 * Converts an input object into datetime metadata. The input
 * may be either a string or a tuple.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_pyobject_to_datetime_metadata(PyObject *obj,
                                      PyArray_DatetimeMetaData *out_meta)
{
    PyObject *ascii = NULL;
    char *str = NULL;
    Py_ssize_t len = 0;

    if (PyTuple_Check(obj)) {
        return convert_datetime_metadata_tuple_to_datetime_metadata(obj,
                                                                out_meta);
    }

    /* Get an ASCII string */
    if (PyUnicode_Check(obj)) {
        /* Allow unicode format strings: convert to bytes */
        ascii = PyUnicode_AsASCIIString(obj);
        if (ascii == NULL) {
            return -1;
        }
    }
    else if (PyBytes_Check(obj)) {
        ascii = obj;
        Py_INCREF(ascii);
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "Invalid object for specifying NumPy datetime metadata");
        return -1;
    }

    if (PyBytes_AsStringAndSize(ascii, &str, &len) < 0) {
        return -1;
    }

    if (len > 0 && str[0] == '[') {
        return parse_datetime_metadata_from_metastr(str, len, out_meta);
    }
    else {
        if (parse_datetime_extended_unit_from_string(str, len,
                                                NULL, out_meta) < 0) {
            return -1;
        }

        return 0;
    }

}

/*
 * 'ret' is a PyUString containing the datetime string, and this
 * function appends the metadata string to it.
 *
 * If 'skip_brackets' is true, skips the '[]'.
 *
 * This function steals the reference 'ret'
 */
NPY_NO_EXPORT PyObject *
append_metastr_to_string(PyArray_DatetimeMetaData *meta,
                                    int skip_brackets,
                                    PyObject *ret)
{
    PyObject *res;
    int num;
    char *basestr;

    if (ret == NULL) {
        return NULL;
    }

    if (meta->base == NPY_FR_GENERIC) {
        /* Without brackets, give a string "generic" */
        if (skip_brackets) {
            PyUString_ConcatAndDel(&ret, PyUString_FromString("generic"));
            return ret;
        }
        /* But with brackets, append nothing */
        else {
            return ret;
        }
    }

    num = meta->num;
    if (meta->base >= 0 && meta->base < NPY_DATETIME_NUMUNITS) {
        basestr = _datetime_strings[meta->base];
    }
    else {
        PyErr_SetString(PyExc_RuntimeError,
                "NumPy datetime metadata is corrupted");
        return NULL;
    }

    if (num == 1) {
        if (skip_brackets) {
            res = PyUString_FromFormat("%s", basestr);
        }
        else {
            res = PyUString_FromFormat("[%s]", basestr);
        }
    }
    else {
        if (skip_brackets) {
            res = PyUString_FromFormat("%d%s", num, basestr);
        }
        else {
            res = PyUString_FromFormat("[%d%s]", num, basestr);
        }
    }

    PyUString_ConcatAndDel(&ret, res);
    return ret;
}

/*
 * Adjusts a datetimestruct based on a seconds offset. Assumes
 * the current values are valid.
 */
NPY_NO_EXPORT void
add_seconds_to_datetimestruct(npy_datetimestruct *dts, int seconds)
{
    int minutes;

    dts->sec += seconds;
    if (dts->sec < 0) {
        minutes = dts->sec / 60;
        dts->sec = dts->sec % 60;
        if (dts->sec < 0) {
            --minutes;
            dts->sec += 60;
        }
        add_minutes_to_datetimestruct(dts, minutes);
    }
    else if (dts->sec >= 60) {
        minutes = dts->sec / 60;
        dts->sec = dts->sec % 60;
        add_minutes_to_datetimestruct(dts, minutes);
    }
}

/*
 * Adjusts a datetimestruct based on a minutes offset. Assumes
 * the current values are valid.
 */
NPY_NO_EXPORT void
add_minutes_to_datetimestruct(npy_datetimestruct *dts, int minutes)
{
    int isleap;

    /* MINUTES */
    dts->min += minutes;
    while (dts->min < 0) {
        dts->min += 60;
        dts->hour--;
    }
    while (dts->min >= 60) {
        dts->min -= 60;
        dts->hour++;
    }

    /* HOURS */
    while (dts->hour < 0) {
        dts->hour += 24;
        dts->day--;
    }
    while (dts->hour >= 24) {
        dts->hour -= 24;
        dts->day++;
    }

    /* DAYS */
    if (dts->day < 1) {
        dts->month--;
        if (dts->month < 1) {
            dts->year--;
            dts->month = 12;
        }
        isleap = is_leapyear(dts->year);
        dts->day += _days_per_month_table[isleap][dts->month-1];
    }
    else if (dts->day > 28) {
        isleap = is_leapyear(dts->year);
        if (dts->day > _days_per_month_table[isleap][dts->month-1]) {
            dts->day -= _days_per_month_table[isleap][dts->month-1];
            dts->month++;
            if (dts->month > 12) {
                dts->year++;
                dts->month = 1;
            }
        }
    }
}

/*
 * Tests for and converts a Python datetime.datetime or datetime.date
 * object into a NumPy npy_datetimestruct.
 *
 * While the C API has PyDate_* and PyDateTime_* functions, the following
 * implementation just asks for attributes, and thus supports
 * datetime duck typing. The tzinfo time zone conversion would require
 * this style of access anyway.
 *
 * 'out_bestunit' gives a suggested unit based on whether the object
 *      was a datetime.date or datetime.datetime object.
 *
 * If 'apply_tzinfo' is 1, this function uses the tzinfo to convert
 * to UTC time, otherwise it returns the struct with the local time.
 *
 * Returns -1 on error, 0 on success, and 1 (with no error set)
 * if obj doesn't have the neeeded date or datetime attributes.
 */
NPY_NO_EXPORT int
convert_pydatetime_to_datetimestruct(PyObject *obj, npy_datetimestruct *out,
                                     NPY_DATETIMEUNIT *out_bestunit,
                                     int apply_tzinfo)
{
    PyObject *tmp;
    int isleap;

#ifdef DATETIME_DEBUG
    fprintf(stderr, "convert_pydatetime_to_datetimestruct\n");
#endif
    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->month = 1;
    out->day = 1;

    /* Need at least year/month/day attributes */
    if (!PyObject_HasAttrString(obj, "year") ||
            !PyObject_HasAttrString(obj, "month") ||
            !PyObject_HasAttrString(obj, "day")) {
        return 1;
    }

    /* Get the year */
    tmp = PyObject_GetAttrString(obj, "year");
    if (tmp == NULL) {
        return -1;
    }
    out->year = PyInt_AsLong(tmp);
    if (out->year == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the month */
    tmp = PyObject_GetAttrString(obj, "month");
    if (tmp == NULL) {
        return -1;
    }
    out->month = PyInt_AsLong(tmp);
    if (out->month == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the day */
    tmp = PyObject_GetAttrString(obj, "day");
    if (tmp == NULL) {
        return -1;
    }
    out->day = PyInt_AsLong(tmp);
    if (out->day == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Validate that the month and day are valid for the year */
    if (out->month < 1 || out->month > 12) {
        goto invalid_date;
    }
    isleap = is_leapyear(out->year);
    if (out->day < 1 ||
                out->day > _days_per_month_table[isleap][out->month-1]) {
        goto invalid_date;
    }

    /* Check for time attributes (if not there, return success as a date) */
    if (!PyObject_HasAttrString(obj, "hour") ||
            !PyObject_HasAttrString(obj, "minute") ||
            !PyObject_HasAttrString(obj, "second") ||
            !PyObject_HasAttrString(obj, "microsecond")) {
        /* The best unit for date is 'D' */
        if (out_bestunit != NULL) {
            *out_bestunit = NPY_FR_D;
        }
        return 0;
    }

    /* Get the hour */
    tmp = PyObject_GetAttrString(obj, "hour");
    if (tmp == NULL) {
        return -1;
    }
    out->hour = PyInt_AsLong(tmp);
    if (out->hour == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the minute */
    tmp = PyObject_GetAttrString(obj, "minute");
    if (tmp == NULL) {
        return -1;
    }
    out->min = PyInt_AsLong(tmp);
    if (out->min == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the second */
    tmp = PyObject_GetAttrString(obj, "second");
    if (tmp == NULL) {
        return -1;
    }
    out->sec = PyInt_AsLong(tmp);
    if (out->sec == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the microsecond */
    tmp = PyObject_GetAttrString(obj, "microsecond");
    if (tmp == NULL) {
        return -1;
    }
    out->us = PyInt_AsLong(tmp);
    if (out->us == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    if (out->hour < 0 || out->hour >= 24 ||
            out->min < 0 || out->min >= 60 ||
            out->sec < 0 || out->sec >= 60 ||
            out->us < 0 || out->us >= 1000000) {
        goto invalid_time;
    }

    /* Apply the time zone offset if it exists */
    if (apply_tzinfo && PyObject_HasAttrString(obj, "tzinfo")) {
        tmp = PyObject_GetAttrString(obj, "tzinfo");
        if (tmp == NULL) {
            return -1;
        }
        if (tmp == Py_None) {
            Py_DECREF(tmp);
        }
        else {
            PyObject *offset;
            int seconds_offset, minutes_offset;

            /* The utcoffset function should return a timedelta */
            offset = PyObject_CallMethod(tmp, "utcoffset", "O", obj);
            if (offset == NULL) {
                Py_DECREF(tmp);
                return -1;
            }
            Py_DECREF(tmp);

            /*
             * The timedelta should have a function "total_seconds"
             * which contains the value we want.
             */
            tmp = PyObject_CallMethod(offset, "total_seconds", "");
            if (tmp == NULL) {
                return -1;
            }
            seconds_offset = PyInt_AsLong(tmp);
            if (seconds_offset == -1 && PyErr_Occurred()) {
                Py_DECREF(tmp);
                return -1;
            }
            Py_DECREF(tmp);

            /* Convert to a minutes offset and apply it */
            minutes_offset = seconds_offset / 60;

            add_minutes_to_datetimestruct(out, -minutes_offset);
        }
    }

    /* The resolution of Python's datetime is 'us' */
    if (out_bestunit != NULL) {
        *out_bestunit = NPY_FR_us;
    }

    return 0;

invalid_date:
    PyErr_Format(PyExc_ValueError,
            "Invalid date (%d,%d,%d) when converting to NumPy datetime",
            (int)out->year, (int)out->month, (int)out->day);
    return -1;

invalid_time:
    PyErr_Format(PyExc_ValueError,
            "Invalid time (%d,%d,%d,%d) when converting "
            "to NumPy datetime",
            (int)out->hour, (int)out->min, (int)out->sec, (int)out->us);
    return -1;
}

/*
 * Gets a tzoffset in minutes by calling the fromutc() function on
 * the Python datetime.tzinfo object.
 */
NPY_NO_EXPORT int
get_tzoffset_from_pytzinfo(PyObject *timezone, npy_datetimestruct *dts)
{
    PyObject *dt, *loc_dt;
    npy_datetimestruct loc_dts;

    /* Create a Python datetime to give to the timezone object */
    dt = PyDateTime_FromDateAndTime((int)dts->year, dts->month, dts->day,
                            dts->hour, dts->min, 0, 0);
    if (dt == NULL) {
        return -1;
    }

    /* Convert the datetime from UTC to local time */
    loc_dt = PyObject_CallMethod(timezone, "fromutc", "O", dt);
    Py_DECREF(dt);
    if (loc_dt == NULL) {
        return -1;
    }

    /* Convert the local datetime into a datetimestruct */
    if (convert_pydatetime_to_datetimestruct(loc_dt, &loc_dts, NULL, 0) < 0) {
        Py_DECREF(loc_dt);
        return -1;
    }

    Py_DECREF(loc_dt);

    /* Calculate the tzoffset as the difference between the datetimes */
    return get_datetimestruct_minutes(&loc_dts) -
           get_datetimestruct_minutes(dts);
}

/*
 * Converts a PyObject * into a datetime, in any of the forms supported.
 *
 * If the units metadata isn't known ahead of time, set meta->base
 * to -1, and this function will populate meta with either default
 * values or values from the input object.
 *
 * The 'casting' parameter is used to control what kinds of inputs
 * are accepted, and what happens. For example, with 'unsafe' casting,
 * unrecognized inputs are converted to 'NaT' instead of throwing an error,
 * while with 'safe' casting an error will be thrown if any precision
 * from the input will be thrown away.
 *
 * Returns -1 on error, 0 on success.
 */
NPY_NO_EXPORT int
convert_pyobject_to_datetime(PyArray_DatetimeMetaData *meta, PyObject *obj,
                                NPY_CASTING casting, npy_datetime *out)
{
    if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        PyObject *bytes = NULL;
        char *str = NULL;
        Py_ssize_t len = 0;
        npy_datetimestruct dts;
        NPY_DATETIMEUNIT bestunit = -1;

        /* Convert to an ASCII string for the date parser */
        if (PyUnicode_Check(obj)) {
            bytes = PyUnicode_AsASCIIString(obj);
            if (bytes == NULL) {
                return -1;
            }
        }
        else {
            bytes = obj;
            Py_INCREF(bytes);
        }
        if (PyBytes_AsStringAndSize(bytes, &str, &len) == -1) {
            Py_DECREF(bytes);
            return -1;
        }

        /* Parse the ISO date */
        if (parse_iso_8601_datetime(str, len, meta->base, casting,
                                &dts, NULL, &bestunit, NULL) < 0) {
            Py_DECREF(bytes);
            return -1;
        }
        Py_DECREF(bytes);

        /* Use the detected unit if none was specified */
        if (meta->base == -1) {
            meta->base = bestunit;
            meta->num = 1;
        }

        if (convert_datetimestruct_to_datetime(meta, &dts, out) < 0) {
            return -1;
        }

        return 0;
    }
    /* Do no conversion on raw integers */
    else if (PyInt_Check(obj) || PyLong_Check(obj)) {
        /* Don't allow conversion from an integer without specifying a unit */
        if (meta->base == -1 || meta->base == NPY_FR_GENERIC) {
            PyErr_SetString(PyExc_ValueError, "Converting an integer to a "
                            "NumPy datetime requires a specified unit");
            return -1;
        }
        *out = PyLong_AsLongLong(obj);
        return 0;
    }
    /* Datetime scalar */
    else if (PyArray_IsScalar(obj, Datetime)) {
        PyDatetimeScalarObject *dts = (PyDatetimeScalarObject *)obj;

        /* Copy the scalar directly if units weren't specified */
        if (meta->base == -1) {
            *meta = dts->obmeta;
            *out = dts->obval;

            return 0;
        }
        /* Otherwise do a casting transformation */
        else {
            /* Allow NaT (not-a-time) values to slip through any rule */
            if (dts->obval != NPY_DATETIME_NAT &&
                        raise_if_datetime64_metadata_cast_error(
                                "NumPy timedelta64 scalar",
                                &dts->obmeta, meta, casting) < 0) {
                return -1;
            }
            else {
                return cast_datetime_to_datetime(&dts->obmeta, meta,
                                                    dts->obval, out);
            }
        }
    }
    /* Datetime zero-dimensional array */
    else if (PyArray_Check(obj) &&
                    PyArray_NDIM(obj) == 0 &&
                    PyArray_DESCR(obj)->type_num == NPY_DATETIME) {
        PyArray_DatetimeMetaData *obj_meta;
        npy_datetime dt = 0;

        obj_meta = get_datetime_metadata_from_dtype(PyArray_DESCR(obj));
        if (obj_meta == NULL) {
            return -1;
        }
        PyArray_DESCR(obj)->f->copyswap(&dt,
                                        PyArray_DATA(obj),
                                        !PyArray_ISNOTSWAPPED(obj),
                                        obj);

        /* Copy the value directly if units weren't specified */
        if (meta->base == -1) {
            *meta = *obj_meta;
            *out = dt;

            return 0;
        }
        /* Otherwise do a casting transformation */
        else {
            /* Allow NaT (not-a-time) values to slip through any rule */
            if (dt != NPY_DATETIME_NAT &&
                        raise_if_datetime64_metadata_cast_error(
                                "NumPy timedelta64 scalar",
                                obj_meta, meta, casting) < 0) {
                return -1;
            }
            else {
                return cast_datetime_to_datetime(obj_meta, meta, dt, out);
            }
        }
    }
    /* Convert from a Python date or datetime object */
    else {
        int code;
        npy_datetimestruct dts;
        NPY_DATETIMEUNIT bestunit = -1;

        code = convert_pydatetime_to_datetimestruct(obj, &dts, &bestunit, 1);
        if (code == -1) {
            return -1;
        }
        else if (code == 0) {
            /* Use the detected unit if none was specified */
            if (meta->base == -1) {
                meta->base = bestunit;
                meta->num = 1;
            }
            else {
                PyArray_DatetimeMetaData obj_meta;
                obj_meta.base = bestunit;
                obj_meta.num = 1;

                if (raise_if_datetime64_metadata_cast_error(
                                bestunit == NPY_FR_D ? "datetime.date object"
                                                 : "datetime.datetime object",
                                &obj_meta, meta, casting) < 0) {
                    return -1;
                }
            }

            return convert_datetimestruct_to_datetime(meta, &dts, out);
        }
    }

    /*
     * With unsafe casting, convert unrecognized objects into NaT
     * and with same_kind casting, convert None into NaT
     */
    if (casting == NPY_UNSAFE_CASTING ||
            (obj == Py_None && casting == NPY_SAME_KIND_CASTING)) {
        if (meta->base == -1) {
            meta->base = NPY_FR_GENERIC;
            meta->num = 1;
        }
        *out = NPY_DATETIME_NAT;
        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Could not convert object to NumPy datetime");
        return -1;
    }
}

/*
 * Converts a PyObject * into a timedelta, in any of the forms supported
 *
 * If the units metadata isn't known ahead of time, set meta->base
 * to -1, and this function will populate meta with either default
 * values or values from the input object.
 *
 * The 'casting' parameter is used to control what kinds of inputs
 * are accepted, and what happens. For example, with 'unsafe' casting,
 * unrecognized inputs are converted to 'NaT' instead of throwing an error,
 * while with 'safe' casting an error will be thrown if any precision
 * from the input will be thrown away.
 *
 * Returns -1 on error, 0 on success.
 */
NPY_NO_EXPORT int
convert_pyobject_to_timedelta(PyArray_DatetimeMetaData *meta, PyObject *obj,
                                NPY_CASTING casting, npy_timedelta *out)
{
    if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        PyObject *bytes = NULL;
        char *str = NULL;
        Py_ssize_t len = 0;
        int succeeded = 0;

        /* Convert to an ASCII string for the date parser */
        if (PyUnicode_Check(obj)) {
            bytes = PyUnicode_AsASCIIString(obj);
            if (bytes == NULL) {
                return -1;
            }
        }
        else {
            bytes = obj;
            Py_INCREF(bytes);
        }
        if (PyBytes_AsStringAndSize(bytes, &str, &len) == -1) {
            Py_DECREF(bytes);
            return -1;
        }

        /* Check for a NaT string */
        if (len <= 0 || (len == 3 &&
                        tolower(str[0]) == 'n' &&
                        tolower(str[1]) == 'a' &&
                        tolower(str[2]) == 't')) {
            *out = NPY_DATETIME_NAT;
            succeeded = 1;
        }
        /* Parse as an integer */
        else {
            char *strend = NULL;

            *out = strtol(str, &strend, 10);
            if (strend - str == len) {
                succeeded = 1;
            }
        }

        if (succeeded) {
            /* Use generic units if none was specified */
            if (meta->base == -1) {
                meta->base = NPY_FR_GENERIC;
                meta->num = 1;
            }

            return 0;
        }
    }
    /* Do no conversion on raw integers */
    else if (PyInt_Check(obj) || PyLong_Check(obj)) {
        /* Use the default unit if none was specified */
        if (meta->base == -1) {
            meta->base = NPY_DATETIME_DEFAULTUNIT;
            meta->num = 1;
        }

        *out = PyLong_AsLongLong(obj);
        return 0;
    }
    /* Timedelta scalar */
    else if (PyArray_IsScalar(obj, Timedelta)) {
        PyTimedeltaScalarObject *dts = (PyTimedeltaScalarObject *)obj;

        /* Copy the scalar directly if units weren't specified */
        if (meta->base == -1) {
            *meta = dts->obmeta;
            *out = dts->obval;

            return 0;
        }
        /* Otherwise do a casting transformation */
        else {
            /* Allow NaT (not-a-time) values to slip through any rule */
            if (dts->obval != NPY_DATETIME_NAT &&
                        raise_if_timedelta64_metadata_cast_error(
                                "NumPy timedelta64 scalar",
                                &dts->obmeta, meta, casting) < 0) {
                return -1;
            }
            else {
                return cast_timedelta_to_timedelta(&dts->obmeta, meta,
                                                    dts->obval, out);
            }
        }
    }
    /* Timedelta zero-dimensional array */
    else if (PyArray_Check(obj) &&
                    PyArray_NDIM(obj) == 0 &&
                    PyArray_DESCR(obj)->type_num == NPY_TIMEDELTA) {
        PyArray_DatetimeMetaData *obj_meta;
        npy_timedelta dt = 0;

        obj_meta = get_datetime_metadata_from_dtype(PyArray_DESCR(obj));
        if (obj_meta == NULL) {
            return -1;
        }
        PyArray_DESCR(obj)->f->copyswap(&dt,
                                        PyArray_DATA(obj),
                                        !PyArray_ISNOTSWAPPED(obj),
                                        obj);

        /* Copy the value directly if units weren't specified */
        if (meta->base == -1) {
            *meta = *obj_meta;
            *out = dt;

            return 0;
        }
        /* Otherwise do a casting transformation */
        else {
            /* Allow NaT (not-a-time) values to slip through any rule */
            if (dt != NPY_DATETIME_NAT &&
                        raise_if_timedelta64_metadata_cast_error(
                                "NumPy timedelta64 scalar",
                                obj_meta, meta, casting) < 0) {
                return -1;
            }
            else {
                return cast_timedelta_to_timedelta(obj_meta, meta, dt, out);
            }
        }
    }
    /* Convert from a Python timedelta object */
    else if (PyObject_HasAttrString(obj, "days") &&
                PyObject_HasAttrString(obj, "seconds") &&
                PyObject_HasAttrString(obj, "microseconds")) {
        PyObject *tmp;
        PyArray_DatetimeMetaData us_meta;
        npy_timedelta td;
        npy_int64 days;
        int seconds = 0, useconds = 0;

        /* Get the days */
        tmp = PyObject_GetAttrString(obj, "days");
        if (tmp == NULL) {
            return -1;
        }
        days = PyLong_AsLongLong(tmp);
        if (days == -1 && PyErr_Occurred()) {
            Py_DECREF(tmp);
            return -1;
        }
        Py_DECREF(tmp);

        /* Get the seconds */
        tmp = PyObject_GetAttrString(obj, "seconds");
        if (tmp == NULL) {
            return -1;
        }
        seconds = PyInt_AsLong(tmp);
        if (seconds == -1 && PyErr_Occurred()) {
            Py_DECREF(tmp);
            return -1;
        }
        Py_DECREF(tmp);

        /* Get the microseconds */
        tmp = PyObject_GetAttrString(obj, "microseconds");
        if (tmp == NULL) {
            return -1;
        }
        useconds = PyInt_AsLong(tmp);
        if (useconds == -1 && PyErr_Occurred()) {
            Py_DECREF(tmp);
            return -1;
        }
        Py_DECREF(tmp);

        td = days*(24*60*60*1000000LL) + seconds*1000000LL + useconds;

        /* Use microseconds if none was specified */
        if (meta->base == -1) {
            meta->base = NPY_FR_us;
            meta->num = 1;

            *out = td;

            return 0;
        }
        else {
            /*
             * Detect the largest unit where every value after is zero,
             * to allow safe casting to seconds if microseconds is zero,
             * for instance.
             */
            if (td % 1000LL != 0) {
                us_meta.base = NPY_FR_us;
            }
            else if (td % 1000000LL != 0) {
                us_meta.base = NPY_FR_ms;
            }
            else if (td % (60*1000000LL) != 0) {
                us_meta.base = NPY_FR_s;
            }
            else if (td % (60*60*1000000LL) != 0) {
                us_meta.base = NPY_FR_m;
            }
            else if (td % (24*60*60*1000000LL) != 0) {
                us_meta.base = NPY_FR_D;
            }
            else if (td % (7*24*60*60*1000000LL) != 0) {
                us_meta.base = NPY_FR_W;
            }
            us_meta.num = 1;

            if (raise_if_timedelta64_metadata_cast_error(
                                "datetime.timedelta object",
                                &us_meta, meta, casting) < 0) {
                return -1;
            }
            else {
                /* Switch back to microseconds for the casting operation */
                us_meta.base = NPY_FR_us;

                return cast_timedelta_to_timedelta(&us_meta, meta, td, out);
            }
        }
    }

    /*
     * With unsafe casting, convert unrecognized objects into NaT
     * and with same_kind casting, convert None into NaT
     */
    if (casting == NPY_UNSAFE_CASTING ||
            (obj == Py_None && casting == NPY_SAME_KIND_CASTING)) {
        if (meta->base == -1) {
            meta->base = NPY_FR_GENERIC;
            meta->num = 1;
        }
        *out = NPY_DATETIME_NAT;
        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Could not convert object to NumPy timedelta");
        return -1;
    }
}

/*
 * Converts a datetime into a PyObject *.
 *
 * Not-a-time is returned as the string "NaT".
 * For days or coarser, returns a datetime.date.
 * For microseconds or coarser, returns a datetime.datetime.
 * For units finer than microseconds, returns an integer.
 */
NPY_NO_EXPORT PyObject *
convert_datetime_to_pyobject(npy_datetime dt, PyArray_DatetimeMetaData *meta)
{
    PyObject *ret = NULL;
    npy_datetimestruct dts;
#ifdef DATETIME_DEBUG
    fprintf(stderr, "convert_datetime_to_pyobject: %lld\n", (long long)dt);
#endif

    /*
     * Convert NaT (not-a-time) and any value with generic units
     * into None.
     */
    if (dt == NPY_DATETIME_NAT || meta->base == NPY_FR_GENERIC) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    /* If the type's precision is greater than microseconds, return an int */
    if (meta->base > NPY_FR_us) {
        return PyLong_FromLongLong(dt);
    }

    /* Convert to a datetimestruct */
    if (convert_datetime_to_datetimestruct(meta, dt, &dts) < 0) {
        return NULL;
    }

    /*
     * If the year is outside the range of years supported by Python's
     * datetime, or the datetime64 falls on a leap second,
     * return a raw int.
     */
    if (dts.year < 1 || dts.year > 9999 || dts.sec == 60) {
        return PyLong_FromLongLong(dt);
    }

    /* If the type's precision is greater than days, return a datetime */
    if (meta->base > NPY_FR_D) {
        ret = PyDateTime_FromDateAndTime(dts.year, dts.month, dts.day,
                                dts.hour, dts.min, dts.sec, dts.us);
    }
    /* Otherwise return a date */
    else {
#ifdef DATETIME_DEBUG
        fprintf(stderr, "convert_datetime_to_pyobject: return using PyDate_FromDate\n");
#endif
        ret = PyDate_FromDate(dts.year, dts.month, dts.day);
    }

    return ret;
}

/*
 * Converts a timedelta into a PyObject *.
 *
 * Not-a-time is returned as the string "NaT".
 * For microseconds or coarser, returns a datetime.timedelta.
 * For units finer than microseconds, returns an integer.
 */
NPY_NO_EXPORT PyObject *
convert_timedelta_to_pyobject(npy_timedelta td, PyArray_DatetimeMetaData *meta)
{
    PyObject *ret = NULL;
    npy_timedelta value;
    int days = 0, seconds = 0, useconds = 0;

    /*
     * Convert NaT (not-a-time) into None.
     */
    if (td == NPY_DATETIME_NAT) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    /*
     * If the type's precision is greater than microseconds, is
     * Y/M/B (nonlinear units), or is generic units, return an int
     */
    if (meta->base > NPY_FR_us ||
                    meta->base == NPY_FR_Y ||
                    meta->base == NPY_FR_M ||
                    meta->base == NPY_FR_B ||
                    meta->base == NPY_FR_GENERIC) {
#ifdef DATETIME_DEBUG
        fprintf(stderr, "convert_timedelta_to_pyobject: return %lld\n", (long long)td);
#endif
        return PyLong_FromLongLong(td);
    }

    value = td;

    /* Apply the unit multiplier (TODO: overflow treatment...) */
    value *= meta->num;

    /* Convert to days/seconds/useconds */
    switch (meta->base) {
        case NPY_FR_W:
            value *= 7;
            break;
        case NPY_FR_D:
            break;
        case NPY_FR_h:
            seconds = (int)((value % 24) * (60*60));
            value = value / 24;
            break;
        case NPY_FR_m:
            seconds = (int)(value % (24*60)) * 60;
            value = value / (24*60);
            break;
        case NPY_FR_s:
            seconds = (int)(value % (24*60*60));
            value = value / (24*60*60);
            break;
        case NPY_FR_ms:
            useconds = (int)(value % 1000) * 1000;
            value = value / 1000;
            seconds = (int)(value % (24*60*60));
            value = value / (24*60*60);
            break;
        case NPY_FR_us:
            useconds = (int)(value % (1000*1000));
            value = value / (1000*1000);
            seconds = (int)(value % (24*60*60));
            value = value / (24*60*60);
            break;
        default:
            break;
    }
    /*
     * 'value' represents days, and seconds/useconds are filled.
     *
     * If it would overflow the datetime.timedelta days, return a raw int
     */
    if (value < -999999999 || value > 999999999) {
        return PyLong_FromLongLong(td);
    }
    else {
        days = (int)value;
        ret = PyDelta_FromDSU(days, seconds, useconds);
        if (ret == NULL) {
            return NULL;
        }
    }

    return ret;
}

/*
 * Returns true if the datetime metadata matches
 */
NPY_NO_EXPORT npy_bool
has_equivalent_datetime_metadata(PyArray_Descr *type1, PyArray_Descr *type2)
{
    PyArray_DatetimeMetaData *meta1, *meta2;

    if ((type1->type_num != NPY_DATETIME &&
                        type1->type_num != NPY_TIMEDELTA) ||
                    (type2->type_num != NPY_DATETIME &&
                        type2->type_num != NPY_TIMEDELTA)) {
        return 0;
    }

    meta1 = get_datetime_metadata_from_dtype(type1);
    if (meta1 == NULL) {
        PyErr_Clear();
        return 0;
    }
    meta2 = get_datetime_metadata_from_dtype(type2);
    if (meta2 == NULL) {
        PyErr_Clear();
        return 0;
    }

    /* For generic units, the num is ignored */
    if (meta1->base == NPY_FR_GENERIC && meta2->base == NPY_FR_GENERIC) {
        return 1;
    }

    return meta1->base == meta2->base &&
            meta1->num == meta2->num;
}

/*
 * Casts a single datetime from having src_meta metadata into
 * dst_meta metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
cast_datetime_to_datetime(PyArray_DatetimeMetaData *src_meta,
                          PyArray_DatetimeMetaData *dst_meta,
                          npy_datetime src_dt,
                          npy_datetime *dst_dt)
{
    npy_datetimestruct dts;

#ifdef DATETIME_DEBUG
    fprintf(stderr, "cast_datetime_to_datetime: source = %lld, bases: %d %d\n", (long long)src_dt, src_meta->base, dst_meta->base);
#endif
    /* If the metadata is the same, short-circuit the conversion */
    if (src_meta->base == dst_meta->base &&
            src_meta->num == dst_meta->num) {
        *dst_dt = src_dt;
        return 0;
    }

    /* Otherwise convert through a datetimestruct */
    if (convert_datetime_to_datetimestruct(src_meta, src_dt, &dts) < 0) {
            *dst_dt = NPY_DATETIME_NAT;
            return -1;
    }
    if (convert_datetimestruct_to_datetime(dst_meta, &dts, dst_dt) < 0) {
        *dst_dt = NPY_DATETIME_NAT;
        return -1;
    }

    return 0;
}

/*
 * Casts a single timedelta from having src_meta metadata into
 * dst_meta metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
cast_timedelta_to_timedelta(PyArray_DatetimeMetaData *src_meta,
                          PyArray_DatetimeMetaData *dst_meta,
                          npy_timedelta src_dt,
                          npy_timedelta *dst_dt)
{
    npy_int64 num = 0, denom = 0;

    /* If the metadata is the same, short-circuit the conversion */
    if (src_meta->base == dst_meta->base &&
            src_meta->num == dst_meta->num) {
        *dst_dt = src_dt;
        return 0;
    }

    /* Get the conversion factor */
    get_datetime_conversion_factor(src_meta, dst_meta, &num, &denom);

    if (num == 0) {
        return -1;
    }

    /* Apply the scaling */
    if (src_dt < 0) {
        *dst_dt = (src_dt * num - (denom - 1)) / denom;
    }
    else {
        *dst_dt = src_dt * num / denom;
    }

    return 0;
}

/*
 * Returns true if the object is something that is best considered
 * a Datetime, false otherwise.
 */
static npy_bool
is_any_numpy_datetime(PyObject *obj)
{
    return (PyArray_IsScalar(obj, Datetime) ||
            (PyArray_Check(obj) && (
                PyArray_DESCR(obj)->type_num == NPY_DATETIME)) ||
            PyDate_Check(obj) ||
            PyDateTime_Check(obj));
}

/*
 * Returns true if the object is something that is best considered
 * a Timedelta, false otherwise.
 */
static npy_bool
is_any_numpy_timedelta(PyObject *obj)
{
    return (PyArray_IsScalar(obj, Timedelta) ||
            (PyArray_Check(obj) && (
                PyArray_DESCR(obj)->type_num == NPY_TIMEDELTA)) ||
            PyDelta_Check(obj));
}

/*
 * Returns true if the object is something that is best considered
 * a Datetime or Timedelta, false otherwise.
 */
NPY_NO_EXPORT npy_bool
is_any_numpy_datetime_or_timedelta(PyObject *obj)
{
    return obj != NULL &&
           (is_any_numpy_datetime(obj) ||
            is_any_numpy_timedelta(obj));
}

/*
 * Converts an array of PyObject * into datetimes and/or timedeltas,
 * based on the values in type_nums.
 *
 * If inout_meta->base is -1, uses GCDs to calculate the metadata, filling
 * in 'inout_meta' with the resulting metadata. Otherwise uses the provided
 * 'inout_meta' for all the conversions.
 *
 * When obj[i] is NULL, out_value[i] will be set to NPY_DATETIME_NAT.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_pyobjects_to_datetimes(int count,
                               PyObject **objs, int *type_nums,
                               NPY_CASTING casting,
                               npy_int64 *out_values,
                               PyArray_DatetimeMetaData *inout_meta)
{
    int i, is_out_strict;
    PyArray_DatetimeMetaData *meta;

    /* No values trivially succeeds */
    if (count == 0) {
        return 0;
    }

    /* Use the inputs to resolve the unit metadata if requested */
    if (inout_meta->base == -1) {
        /* Allocate an array of metadata corresponding to the objects */
        meta = PyArray_malloc(count * sizeof(PyArray_DatetimeMetaData));
        if (meta == NULL) {
            PyErr_NoMemory();
            return -1;
        }

        /* Convert all the objects into timedeltas or datetimes */
        for (i = 0; i < count; ++i) {
            meta[i].base = -1;
            meta[i].num = 1;

            /* NULL -> NaT */
            if (objs[i] == NULL) {
                out_values[i] = NPY_DATETIME_NAT;
                meta[i].base = NPY_FR_GENERIC;
            }
            else if (type_nums[i] == NPY_DATETIME) {
                if (convert_pyobject_to_datetime(&meta[i], objs[i],
                                            casting, &out_values[i]) < 0) {
                    PyArray_free(meta);
                    return -1;
                }
            }
            else if (type_nums[i] == NPY_TIMEDELTA) {
                if (convert_pyobject_to_timedelta(&meta[i], objs[i],
                                            casting, &out_values[i]) < 0) {
                    PyArray_free(meta);
                    return -1;
                }
            }
            else {
                PyErr_SetString(PyExc_ValueError,
                        "convert_pyobjects_to_datetimes requires that "
                        "all the type_nums provided be datetime or timedelta");
                PyArray_free(meta);
                return -1;
            }
        }

        /* Merge all the metadatas, starting with the first one */
        *inout_meta = meta[0];
        is_out_strict = (type_nums[0] == NPY_TIMEDELTA);

        for (i = 1; i < count; ++i) {
            if (compute_datetime_metadata_greatest_common_divisor(
                                    &meta[i], inout_meta, inout_meta,
                                    type_nums[i] == NPY_TIMEDELTA,
                                    is_out_strict) < 0) {
                PyArray_free(meta);
                return -1;
            }
            is_out_strict = is_out_strict || (type_nums[i] == NPY_TIMEDELTA);
        }

        /* Convert all the values into the resolved unit metadata */
        for (i = 0; i < count; ++i) {
            if (type_nums[i] == NPY_DATETIME) {
                if (cast_datetime_to_datetime(&meta[i], inout_meta,
                                         out_values[i], &out_values[i]) < 0) {
                    PyArray_free(meta);
                    return -1;
                }
            }
            else if (type_nums[i] == NPY_TIMEDELTA) {
                if (cast_timedelta_to_timedelta(&meta[i], inout_meta,
                                         out_values[i], &out_values[i]) < 0) {
                    PyArray_free(meta);
                    return -1;
                }
            }
        }

        PyArray_free(meta);
    }
    /* Otherwise convert to the provided unit metadata */
    else {
        /* Convert all the objects into timedeltas or datetimes */
        for (i = 0; i < count; ++i) {
            /* NULL -> NaT */
            if (objs[i] == NULL) {
                out_values[i] = NPY_DATETIME_NAT;
            }
            else if (type_nums[i] == NPY_DATETIME) {
                if (convert_pyobject_to_datetime(inout_meta, objs[i],
                                            casting, &out_values[i]) < 0) {
                    return -1;
                }
            }
            else if (type_nums[i] == NPY_TIMEDELTA) {
                if (convert_pyobject_to_timedelta(inout_meta, objs[i],
                                            casting, &out_values[i]) < 0) {
                    return -1;
                }
            }
            else {
                PyErr_SetString(PyExc_ValueError,
                        "convert_pyobjects_to_datetimes requires that "
                        "all the type_nums provided be datetime or timedelta");
                return -1;
            }
        }
    }

    return 0;
}

NPY_NO_EXPORT PyArrayObject *
datetime_arange(PyObject *start, PyObject *stop, PyObject *step,
                PyArray_Descr *dtype)
{
    PyArray_DatetimeMetaData meta;
    /*
     * Both datetime and timedelta are stored as int64, so they can
     * share value variables.
     */
    npy_int64 values[3];
    PyObject *objs[3];
    int type_nums[3];

    npy_intp i, length;
    PyArrayObject *ret;
    npy_int64 *ret_data;

    /*
     * First normalize the input parameters so there is no Py_None,
     * and start is moved to stop if stop is unspecified.
     */
    if (step == Py_None) {
        step = NULL;
    }
    if (stop == NULL || stop == Py_None) {
        stop = start;
        start = NULL;
        /* If start was NULL or None, raise an exception */
        if (stop == NULL || stop == Py_None) {
            PyErr_SetString(PyExc_ValueError,
                    "arange needs at least a stopping value");
            return NULL;
        }
    }
    if (start == Py_None) {
        start = NULL;
    }

    /* Step must not be a Datetime */
    if (step != NULL && is_any_numpy_datetime(step)) {
        PyErr_SetString(PyExc_ValueError,
                    "cannot use a datetime as a step in arange");
        return NULL;
    }

    /* Check if the units of the given dtype are generic, in which
     * case we use the code path that detects the units
     */
    if (dtype != NULL) {
        PyArray_DatetimeMetaData *meta_tmp;

        type_nums[0] = dtype->type_num;
        if (type_nums[0] != NPY_DATETIME && type_nums[0] != NPY_TIMEDELTA) {
            PyErr_SetString(PyExc_ValueError,
                        "datetime_arange was given a non-datetime dtype");
            return NULL;
        }

        meta_tmp = get_datetime_metadata_from_dtype(dtype);
        if (meta_tmp == NULL) {
            return NULL;
        }

        /*
         * If the dtype specified is in generic units, detect the
         * units from the input parameters.
         */
        if (meta_tmp->base == NPY_FR_GENERIC) {
            dtype = NULL;
            meta.base = -1;
        }
        /* Otherwise use the provided metadata */
        else {
            meta = *meta_tmp;
        }
    }
    else {
        if (is_any_numpy_datetime(start) || is_any_numpy_datetime(stop)) {
            type_nums[0] = NPY_DATETIME;
        }
        else {
            type_nums[0] = NPY_TIMEDELTA;
        }

        meta.base = -1;
    }

    if (type_nums[0] == NPY_DATETIME && start == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "arange requires both a start and a stop for "
                "NumPy datetime64 ranges");
        return NULL;
    }

    /* Set up to convert the objects to a common datetime unit metadata */
    objs[0] = start;
    objs[1] = stop;
    objs[2] = step;
    if (type_nums[0] == NPY_TIMEDELTA) {
        type_nums[1] = NPY_TIMEDELTA;
        type_nums[2] = NPY_TIMEDELTA;
    }
    else {
        if (PyInt_Check(objs[1]) ||
                        PyLong_Check(objs[1]) ||
                        PyArray_IsScalar(objs[1], Integer) ||
                        is_any_numpy_timedelta(objs[1])) {
            type_nums[1] = NPY_TIMEDELTA;
        }
        else {
            type_nums[1] = NPY_DATETIME;
        }
        type_nums[2] = NPY_TIMEDELTA;
    }

    /* Convert all the arguments */
    if (convert_pyobjects_to_datetimes(3, objs, type_nums,
                                NPY_SAME_KIND_CASTING, values, &meta) < 0) {
        return NULL;
    }

    /* If no step was provided, default to 1 */
    if (step == NULL) {
        values[2] = 1;
    }

    /*
     * In the case of arange(datetime, timedelta), convert
     * the timedelta into a datetime by adding the start datetime.
     */
    if (type_nums[0] == NPY_DATETIME && type_nums[1] == NPY_TIMEDELTA) {
        values[1] += values[0];
    }

    /* Now start, stop, and step have their values and matching metadata */
    if (values[0] == NPY_DATETIME_NAT ||
                    values[1] == NPY_DATETIME_NAT ||
                    values[2] == NPY_DATETIME_NAT) {
        PyErr_SetString(PyExc_ValueError,
                    "arange: cannot use NaT (not-a-time) datetime values");
        return NULL;
    }

    /* Calculate the array length */
    if (values[2] > 0 && values[1] > values[0]) {
        length = (values[1] - values[0] + (values[2] - 1)) / values[2];
    }
    else if (values[2] < 0 && values[1] < values[0]) {
        length = (values[1] - values[0] + (values[2] + 1)) / values[2];
    }
    else if (values[2] != 0) {
        length = 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                    "arange: step cannot be zero");
        return NULL;
    }

    /* Create the dtype of the result */
    if (dtype != NULL) {
        Py_INCREF(dtype);
    }
    else {
        dtype = create_datetime_dtype(type_nums[0], &meta);
        if (dtype == NULL) {
            return NULL;
        }
    }

    /* Create the result array */
    ret = (PyArrayObject *)PyArray_NewFromDescr(
                            &PyArray_Type, dtype, 1, &length, NULL,
                            NULL, 0, NULL);
    if (ret == NULL) {
        return NULL;
    }

    if (length > 0) {
        /* Extract the data pointer */
        ret_data = (npy_int64 *)PyArray_DATA(ret);

        /* Create the timedeltas or datetimes */
        for (i = 0; i < length; ++i) {
            *ret_data = values[0];
            values[0] += values[2];
            ret_data++;
        }
    }

    return ret;
}

/*
 * Examines all the strings in the given string array, and parses them
 * to find the right metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
find_string_array_datetime64_type(PyObject *obj,
                        PyArray_DatetimeMetaData *meta)
{
    NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char **dataptr;
    npy_intp *strideptr, *innersizeptr;
    PyArray_Descr *string_dtype;
    int maxlen;
    char *tmp_buffer = NULL;

    npy_datetimestruct dts;
    PyArray_DatetimeMetaData tmp_meta;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(obj) == 0) {
        return 0;
    }

    string_dtype = PyArray_DescrFromType(NPY_STRING);
    if (string_dtype == NULL) {
        return -1;
    }

    /* Use unsafe casting to allow unicode -> ascii string */
    iter = NpyIter_New((PyArrayObject *)obj,
                            NPY_ITER_READONLY|
                            NPY_ITER_EXTERNAL_LOOP|
                            NPY_ITER_BUFFERED,
                        NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                        string_dtype);
    Py_DECREF(string_dtype);
    if (iter == NULL) {
        return -1;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return -1;
    }
    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    /* Get the resulting string length */
    maxlen = NpyIter_GetDescrArray(iter)[0]->elsize;

    /* Allocate a buffer for strings which fill the buffer completely */
    tmp_buffer = PyArray_malloc(maxlen+1);
    if (tmp_buffer == NULL) {
        PyErr_NoMemory();
        NpyIter_Deallocate(iter);
        return -1;
    }

    /* The iteration loop */
    do {
        /* Get the inner loop data/stride/count values */
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;
        char *tmp;

        /* The inner loop */
        while (count--) {
            /* Replicating strnlen with memchr, because Mac OS X lacks it */
            tmp = memchr(data, '\0', maxlen);

            /* If the string is all full, use the buffer */
            if (tmp == NULL) {
                memcpy(tmp_buffer, data, maxlen);
                tmp_buffer[maxlen] = '\0';

                tmp_meta.base = -1;
                if (parse_iso_8601_datetime(tmp_buffer, maxlen, -1,
                                    NPY_UNSAFE_CASTING, &dts, NULL,
                                    &tmp_meta.base, NULL) < 0) {
                    goto fail;
                }
            }
            /* Otherwise parse the data in place */
            else {
                tmp_meta.base = -1;
                if (parse_iso_8601_datetime(data, tmp - data, -1,
                                    NPY_UNSAFE_CASTING, &dts, NULL,
                                    &tmp_meta.base, NULL) < 0) {
                    goto fail;
                }
            }

            tmp_meta.num = 1;
            /* Combine it with 'meta' */
            if (compute_datetime_metadata_greatest_common_divisor(meta,
                            &tmp_meta, meta, 0, 0) < 0) {
                goto fail;
            }


            data += stride;
        }
    } while(iternext(iter));

    PyArray_free(tmp_buffer);
    NpyIter_Deallocate(iter);

    return 0;

fail:
    if (tmp_buffer != NULL) {
        PyArray_free(tmp_buffer);
    }
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    return -1;
}


/*
 * Recursively determines the metadata for an NPY_DATETIME dtype.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
recursive_find_object_datetime64_type(PyObject *obj,
                        PyArray_DatetimeMetaData *meta)
{
    /* Array -> use its metadata */
    if (PyArray_Check(obj)) {
        PyArray_Descr *obj_dtype = PyArray_DESCR(obj);

        if (obj_dtype->type_num == NPY_STRING ||
                            obj_dtype->type_num == NPY_UNICODE) {
            return find_string_array_datetime64_type(obj, meta);
        }
        /* If the array has metadata, use it */
        else if (obj_dtype->type_num == NPY_DATETIME ||
                    obj_dtype->type_num == NPY_TIMEDELTA) {
            PyArray_DatetimeMetaData *tmp_meta;

            /* Get the metadata from the type */
            tmp_meta = get_datetime_metadata_from_dtype(obj_dtype);
            if (tmp_meta == NULL) {
                return -1;
            }

            /* Combine it with 'meta' */
            if (compute_datetime_metadata_greatest_common_divisor(meta,
                            tmp_meta, meta, 0, 0) < 0) {
                return -1;
            }

            return 0;
        }
        /* If it's not an object array, stop looking */
        else if (obj_dtype->type_num != NPY_OBJECT) {
            return 0;
        }
    }
    /* Datetime scalar -> use its metadata */
    else if (PyArray_IsScalar(obj, Datetime)) {
        PyDatetimeScalarObject *dts = (PyDatetimeScalarObject *)obj;

        /* Combine it with 'meta' */
        if (compute_datetime_metadata_greatest_common_divisor(meta,
                        &dts->obmeta, meta, 0, 0) < 0) {
            return -1;
        }

        return 0;
    }
    /* String -> parse it to find out */
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        npy_datetime tmp = 0;
        PyArray_DatetimeMetaData tmp_meta;

        tmp_meta.base = -1;
        tmp_meta.num = 1;

        if (convert_pyobject_to_datetime(&tmp_meta, obj,
                                        NPY_UNSAFE_CASTING, &tmp) < 0) {
            /* If it's a value error, clear the error */
            if (PyErr_Occurred() &&
                    PyErr_GivenExceptionMatches(PyErr_Occurred(),
                                    PyExc_ValueError)) {
                PyErr_Clear();
                return 0;
            }
            /* Otherwise propagate the error */
            else {
                return -1;
            }
        }

        /* Combine it with 'meta' */
        if (compute_datetime_metadata_greatest_common_divisor(meta,
                        &tmp_meta, meta, 0, 0) < 0) {
            return -1;
        }

        return 0;
    }
    /* Python date object -> 'D' */
    else if (PyDate_Check(obj)) {
        PyArray_DatetimeMetaData tmp_meta;

        tmp_meta.base = NPY_FR_D;
        tmp_meta.num = 1;

        /* Combine it with 'meta' */
        if (compute_datetime_metadata_greatest_common_divisor(meta,
                        &tmp_meta, meta, 0, 0) < 0) {
            return -1;
        }

        return 0;
    }
    /* Python datetime object -> 'us' */
    else if (PyDateTime_Check(obj)) {
        PyArray_DatetimeMetaData tmp_meta;

        tmp_meta.base = NPY_FR_us;
        tmp_meta.num = 1;

        /* Combine it with 'meta' */
        if (compute_datetime_metadata_greatest_common_divisor(meta,
                        &tmp_meta, meta, 0, 0) < 0) {
            return -1;
        }

        return 0;
    }

    /* Now check if what we have left is a sequence for recursion */
    if (PySequence_Check(obj)) {
        Py_ssize_t i, len = PySequence_Size(obj);
        if (len < 0 && PyErr_Occurred()) {
            return -1;
        }

        for (i = 0; i < len; ++i) {
            PyObject *f = PySequence_GetItem(obj, i);
            if (f == NULL) {
                return -1;
            }
            if (f == obj) {
                Py_DECREF(f);
                return 0;
            }
            if (recursive_find_object_datetime64_type(f, meta) < 0) {
                Py_DECREF(f);
                return -1;
            }
            Py_DECREF(f);
        }

        return 0;
    }
    /* Otherwise ignore it */
    else {
        return 0;
    }
}

/*
 * Recursively determines the metadata for an NPY_TIMEDELTA dtype.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
recursive_find_object_timedelta64_type(PyObject *obj,
                        PyArray_DatetimeMetaData *meta)
{
    /* Array -> use its metadata */
    if (PyArray_Check(obj)) {
        PyArray_Descr *obj_dtype = PyArray_DESCR(obj);

        /* If the array has metadata, use it */
        if (obj_dtype->type_num == NPY_DATETIME ||
                    obj_dtype->type_num == NPY_TIMEDELTA) {
            PyArray_DatetimeMetaData *tmp_meta;

            /* Get the metadata from the type */
            tmp_meta = get_datetime_metadata_from_dtype(obj_dtype);
            if (tmp_meta == NULL) {
                return -1;
            }

            /* Combine it with 'meta' */
            if (compute_datetime_metadata_greatest_common_divisor(meta,
                            tmp_meta, meta, 0, 0) < 0) {
                return -1;
            }

            return 0;
        }
        /* If it's not an object array, stop looking */
        else if (obj_dtype->type_num != NPY_OBJECT) {
            return 0;
        }
    }
    /* Datetime scalar -> use its metadata */
    else if (PyArray_IsScalar(obj, Timedelta)) {
        PyTimedeltaScalarObject *dts = (PyTimedeltaScalarObject *)obj;

        /* Combine it with 'meta' */
        if (compute_datetime_metadata_greatest_common_divisor(meta,
                        &dts->obmeta, meta, 1, 1) < 0) {
            return -1;
        }

        return 0;
    }
    /* String -> parse it to find out */
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /* No timedelta parser yet */
        return 0;
    }
    /* Python timedelta object -> 'us' */
    else if (PyDelta_Check(obj)) {
        PyArray_DatetimeMetaData tmp_meta;

        tmp_meta.base = NPY_FR_us;
        tmp_meta.num = 1;

        /* Combine it with 'meta' */
        if (compute_datetime_metadata_greatest_common_divisor(meta,
                        &tmp_meta, meta, 0, 0) < 0) {
            return -1;
        }

        return 0;
    }

    /* Now check if what we have left is a sequence for recursion */
    if (PySequence_Check(obj)) {
        Py_ssize_t i, len = PySequence_Size(obj);
        if (len < 0 && PyErr_Occurred()) {
            return -1;
        }

        for (i = 0; i < len; ++i) {
            PyObject *f = PySequence_GetItem(obj, i);
            if (f == NULL) {
                return -1;
            }
            if (f == obj) {
                Py_DECREF(f);
                return 0;
            }
            if (recursive_find_object_timedelta64_type(f, meta) < 0) {
                Py_DECREF(f);
                return -1;
            }
            Py_DECREF(f);
        }

        return 0;
    }
    /* Otherwise ignore it */
    else {
        return 0;
    }
}

/*
 * Examines all the objects in the given Python object by
 * recursively descending the sequence structure. Returns a
 * datetime or timedelta type with metadata based on the data.
 */
NPY_NO_EXPORT PyArray_Descr *
find_object_datetime_type(PyObject *obj, int type_num)
{
    PyArray_DatetimeMetaData meta;

    meta.base = NPY_FR_GENERIC;
    meta.num = 1;

    if (type_num == NPY_DATETIME) {
        if (recursive_find_object_datetime64_type(obj, &meta) < 0) {
            return NULL;
        }
        else {
            return create_datetime_dtype(type_num, &meta);
        }
    }
    else if (type_num == NPY_TIMEDELTA) {
        if (recursive_find_object_timedelta64_type(obj, &meta) < 0) {
            return NULL;
        }
        else {
            return create_datetime_dtype(type_num, &meta);
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                    "find_object_datetime_type needs a datetime or "
                    "timedelta type number");
        return NULL;
    }
}
