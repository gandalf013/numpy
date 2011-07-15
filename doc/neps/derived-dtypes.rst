======================================
A proposal for derived dtypes in NumPy
======================================

:Author: Alok Singhal <alok@gmail.com>
:Author: Mark Wiebe <mwwiebe@gmail.com>
:Date: 2011-07-15

Abstract
========

NumPy has a powerful dtype system, which allows one to create a wide variety of
array types.  Combined with the UFunc mechanism, this lets users do fast
operations on arrays.  In many contexts, it would be useful to work with
data as if were a different dtype than the underlying storage. For example,
a structured dtype representing 2D cartesian coordinates 'x' and 'y' could
be viewed as another structured dtype with polar coordinates 'r' and 'theta',
converting coordinate systems on the fly.

We are proposing an extension to NumPy's dtypes to let users create derived
dtypes from a base dtype.  These derived dtypes allow one to do powerful things,
such as:

- reinterpreting a NumPy array with elements of size :math:`s_1` as another
  array with elements of size :math:`s_2`, where :math:`s_1 \ne s_2`, possibly
  after some transformation.  This is analogous to ``np.view()``, but
  reinterprets the values in a flexible way instead of just looking at the
  low-level binary representation differently.

- Selectively overriding UFuncs, based on type signatures involving the
  derived dtype.  This means that one can write a custom set of UFuncs
  for a derived dtype and use the powerful NumPy UFunc machinery for
  fast computation on arrays of a derived type. While the custom overloads
  are written in Python, they operate on buffer-sized blocks at a time,
  amortizing the Python overhead.

Example Use Case: Excel Serial Dates
====================================

The *Microsoft excel 1900 date system* stores dates as number of
days since January 1, 1900 as a 32-bit integer.  To have a NumPy array of
numbers representing dates in that system, and have it act like an array of
64-bit numpy.datetime64('M8[D]') values, one needs to transform the 32-bit
values to be converted to appropriate 64-bit values.  With derived dtypes, it
would be possible to do so without the need for creating a second NumPy array of
dtype numpy.datetime64('M8[D]').  In this case, the Python code might look
like::

    class excelserial(np.dtype_derived):
        base_type = np.dtype('i4')
        view_dtype = np.dtype('M8[D]')

        def getter(self, x, out):
            out[...] = np.datetime64('1899-12-30') + x

        def setter(self, x, out):
            out[...] = x - np.datetime64('1899-12-30')

    # January 1, 1900 to January 9 1900
    data = numpy.arange(10, dtype='i4')

    # Same, but interpreted as np.datetime64('M8[D]')
    data = numpy.arange(10, dtype=excelserial)
    # Equivalently,
    data = numpy.arange(10, dtype='excelserial')

The preceding is a WIP design, it might be better to do::

    class ExcelSerial...
        ...
    excelserial = ExcelSerial()

Example Use Case: Dividing Dates into Periods
=============================================

Let's say we want create a timeseries of events during a day.  We would like to
record 50 events per day.  For this, we would like to express the times as
records of (date, period) tuples, with the additional constraint that adding 1
period to the last period of a date results in the first period of the next
date::

    (d, 49) + (0, 1) = (d+1, 0)

In addition, instead of forcing the number of periods per day to be 50, we would
like to be able to easily have multiple such dtypes with different number of
periods per day.

In this case, we could create a dtype like this::

    class PeriodDay(np.dtype_derived):
        # The base type and the view type are the same
        base_type = np.dtype([('date', 'M8[D]'), ('period', 'i8')])

        def __init__(self, nperiods):
            np.dtype_derived.__init__(self)
            self.nperiods = nperiods

        # Now, we override some UFuncs to let us do arithmetic on PeriodDay arrays.
        # For example, to add an integer to a PeriodDay, we would write:
        @out('PeriodDay')             # Output dtype of the UFunc
        @in('PeriodDay', 'i8')        # Input dtypes of the UFunc
        @np.dtype_ufunc('add')        # The UFunc being overridden.
        def add_i8(self, x, y, out=None):
            if out is None:
                out = np.empty_like_broadcast([x, y], dtype='PeriodDay')
            out['date'] = x['date'] + y // self.nperiods
            out['period'] = x['period'] + y % self.nperiods
            if out['period'] >= self.nperiods:
                out['period'] -= self.nperiods
                out['date'] += 1

        # Subtract an i8 from a PeriodDay:
        @out('PeriodDay')
        @in('PeriodDay', 'i8')
        @np.dtype_ufunc('subtract')
        def sub_i8(self, x, y, out=None):
            if out is None:
                out = np.empty_like_broadcast([x, y], dtype='PeriodDay')
            out['date'] = x['date'] - y // self.nperiods
            out['period'] = x['period'] - y % self.nperiods
            if out['period'] < self.nperiods:
                out['period'] += self.nperiods
                out['date'] -= 1

        # Subtract PeriodDay from PeriodDay
        @out('i8')
        @in('PeriodDay', 'PeriodDay')
        @np.dtype_ufunc('subtract')
        def sub_i8(self, x, y, out=None):
            if out is None:
                out = np.empty_like_broadcast([x, y], dtype='i8')

            out[...] = (x['date'] - y['date']) * self.nperiods + (x['period'] - y['period'])

    periodday = PeriodDay(50)

    data = np.array(..., dtype=periodday)
