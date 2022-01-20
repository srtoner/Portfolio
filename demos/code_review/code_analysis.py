#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# # Code Review & Timing
# *Part of course curriculum for Stats 507* <br/>
# *Work produced by Stephen Toner, Fall 2021*

import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.stats as stats
import timeit
import pickle
from IPython.core.display import display, HTML

# # Code Review

# The code below takes a list of tuples as its argument, which in the example given are tuples of length 3.
# Then, for each unique value in position 0, the function collects the tuple
# with the greatest value in the last position.

# + eval = False
# sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]
# op = []
# for m in range(len(sample_list)):
#     li = [sample_list[m]]
#     for n in range(len(sample_list)):
#         if (sample_list[m][0] == sample_list[n][0] and
#             sample_list[m][3] != sample_list[n][3]):
#             li.append(sample_list[n])
#         op.append(sorted(li, key=lambda dd: dd[3], reverse=True)[0])
# res = list(set(op))
# -

# ### Commentary: 
# * First and foremost, the code has an index out of bounds error that needed
# to be fixed. Ideally, this can be avoided by parameterizing the `0` and `2`
# arguments with more meaningful variable names, `unique_pos` and `max_pos` for
# unique position and max position, respectively. This makes the code more flexible
# and allows us to introduce error checking to ensure that the 
# variables make sense given the input.
# * In general, the variable names could be more informative; having
# slightly longer variable names (e.g., `output` rather than `op`) is a small
# price to pay for clarity. 
# * Readability can be improved further using "Pythonic" features such as list
# comprehension and adding informative comments.
# * Lastly, we can use a dictionary to create an easily
# accessible data structure from which we can retrieve the tuples we need.

# # List of Tuples

# We wish to generate a random list of n k-tuples of integers between `low` and
# `high`. Before digging in, we need assurance that our inputs are valid.
#
# Error checking of inputs is a best practice for all programming.
# For the next series of functions, we define two helper functions
# to sanitize input, verifying that the arguments passed are 
# positive and non-negative integers, respectively.

# +
def pn_check(n):

    """
    Checks if n is a positive integer and raise appropriate errors

    Parameters
    ----------
    n : int
        Some positive integer

    Raises
    ------
    TypeError
        Raises if n is not an int
    ValueError
        Raises if n is not positive

    Returns
    -------
    None.
    """
    
    try:
        assert(isinstance(n, int))
    except:
        raise TypeError("n must be a integer")
    try:
        assert(n > 0)
    except:
         raise ValueError("n must be positive")
    return

def nn_check(n):
    """
    Checks if n is a non-negative integer and raise appropriate errors

    Parameters
    ----------
    n : int
        Some non-negative integer

    Raises
    ------
    TypeError
        Raises if n is not an int
    ValueError
        Raises if n is not positive

    Returns
    -------
    None.
    """
    
    try:
        assert(isinstance(n, int))
    except:
        raise TypeError("n must be a integer")
    try:
        assert(n >= 0)
    except:
         raise ValueError("n must be non-negative")
    return
# -

# Now we can conduct these logical checks at the start of a function that uses
# `np.randint` to generate a random list of n k-tuples.

def gen_rand_tuples(n, k, low, high):
    """
    Parameters
    ----------
    n : int
        number of tuples to return
    k : int
        length of tuples to return
    low : int
        lower bound of the range from which we select random integers
    high :
        upper bound of the range from which we select random integers

    Raises
    ------
    TypeError
        Raised if low is not less than high
    ValueError
        Raised if either one of low, high are not integers

    Returns
    -------
    tuple_list : list of int tuples
        returns randomly generated list of n k-tuples
    """
    
    pn_check(n)
    pn_check(k)
    
    try:
        assert(isinstance(low, int))
        assert(isinstance(high, int))
    except:
        raise TypeError("Both low and high must be integers")
        
    try:
        assert(low <= high)
    except:
        raise ValueError("low must be less than or equal to high")
  
    tuple_list = []
    
    for i in np.random.randint(low,high, size = (n, k)):
        tuple_list.append(tuple(i))
    
    assert(all(isinstance(elt, tuple) for elt in tuple_list))
    return tuple_list

# # Refactor the Code Snippet

# We encapsulate the code in a function and give it a more meaningful name, 
# `unique_val_max_pos`. For now we make no changes to the code aside from
# those required to have it run correctly.

def unique_val_max_pos(sample_list, unique_pos, max_pos):
    """
    Parameters
    ----------
    sample_list : list of int tuples
        Input list for Unique Value, Max Position procedure
    unique_pos : int
        Position for which we seek a unique value
    max_pos : int
        The position for which we want the maximal value

    Returns
    -------
    TYPE
        List of tuples with unique values in unique_pos and maximal values in
        max_pos

    """
    op = []
    for m in range(len(sample_list)): 
        li = [sample_list[m]] 
        for n in range(len(sample_list)):
            if (sample_list[m][unique_pos] == sample_list[n][unique_pos] and
                sample_list[m][max_pos] != sample_list[n][max_pos]):
                li.append(sample_list[n])
        op.append(sorted(li, key=lambda dd: dd[max_pos], reverse=True)[0])
    
    return list(set(op))

# While abbreviations can often create confusion,`unique_val_max_pos`
# is a bit too long for us to use in code. For the next series of functions,
# I'll use the abbreviation `uvmp` as a suffix and have comments to ensure the
# meaning is still clear.

# Let's try to improve `unique_val_max_pos` by doing the
# following:
# * Add error checking to prevent index out-of-bounds and other errors
# * Create more meaningful variable names
# * Use list comprehension and a `defaultdict` to improve readability

# First, we declare the function `logic_check`, which performs a suite
# of sanity checks to validate the inputs passed to the variations of
# code we just reviewed.

def logic_check(input_list, u_pos, max_pos):
    """

    Parameters
    ----------
    input_list : list of tuples of integers
       input list to sanitize for "Unique Value, Max Position" procedure
    u_pos : int
        The position which will be used as a unique id for all the tuples
    max_pos : int
        The position for which we seek the maximal value for a unique id

    Raises
    ------
    ValueError
        ValueError with informative comment of why the input is invalid
        While we could check every element of every tuple to ensure that it
        is an int, doing so would be prohibitively inefficient and tedious.
        We need to trust that we are not given invalid input

    Returns
    -------
    None, raises assertions for clearly invalid input

    """
    l_check = False
    nn_check(u_pos)
    nn_check(max_pos)
    try:
        assert(len(input_list) > 0)
    except:
        raise ValueError("input_list is empty")
        
    try:
        assert(isinstance(input_list, list))
    except:
        raise ValueError("input_list must be a list")
    
    min_len =  len(input_list[0])
    
    for i in input_list:
        try:
            assert(isinstance(i, tuple))
        except:
            raise ValueError("input_list must be a list of tuples")
        if(len(i) <= min_len):
            min_len = len(i)
        try:
            assert(np.issubdtype(i[min_len - 1], np.int64) or 
                   np.issubdtype(i[min_len - 1], np.int32))
        except:
            print(type(i[min_len - 1]))
            raise ValueError("input_list must be a list of tuples of integers")
            
    l_check = ((u_pos < min_len) and (max_pos < min_len))
    
    try:
        assert(l_check)
    except:
        raise ValueError("u_pos or max_pos is greater than the size" +
                         " of smallest tuple passed")
    return 

def dict_uvmp(input_list, unique_pos, max_pos):
    """
    Parameters
    ----------
    input_list : list of int tuples
        Input list for Unique Value, Max Position procedure
    unique_pos : int
        Position for which we seek a unique value
    max_pos : int
        The position for which we want the maximal value

    Returns
    -------
    TYPE
        List of tuples with unique values in unique_pos and maximal values in
        max_pos

    """
    logic_check(input_list, unique_pos, max_pos)
    
    output = set()
    results = defaultdict(list)
    
    for x in input_list:
        if len(x):
            results[x[unique_pos]].append(x)
    for y in list(results):
        output.add(max(results[y], key=lambda z: z[max_pos]))
    return list(output)
 
# While this new function is a great improvement, it still suffers from
# inefficiency because it iterates once over `input_list`, and then again
# over the dictionary of results to find the tuple with the greatest value in
# `max_pos`. In the worst case where each of the n tuples has a unique value at
# `unique_pos`, this is $O(n^2)$. We can save time by sorting `input_list`
# by `unique_pos` at the beginning, and only adding a tuple to the output 
# list if it has a value at `max_pos` greater than what we've already seen.

def improved_uvmp(input_list, unique_pos, max_pos):
    """
    Parameters
    ----------
    input_list : list of int tuples
        Input list for Unique Value, Max Position procedure
    unique_pos : int
        Position for which we seek a unique value
    max_pos : int
        The position for which we want the maximal value

    Returns
    -------
    TYPE
        List of tuples with unique values in unique_pos and maximal values in
        max_pos

    """
    logic_check(input_list, unique_pos, max_pos)
    output = []
    
    last_val = None
    
    sorted_list = sorted(input_list, key=lambda up: up[unique_pos])

    for m in sorted_list:
        if m[unique_pos] != last_val:
            last_val = m[unique_pos]
            output.append(m)
        
        if output[-1][max_pos] < m[max_pos]:
            output.pop()
            output.append(m)
    return output

# #### Monte Carlo Runtime Study
# We can use the `gen_rand_tuples` function to 
# conduct a Monte Carlo study of the performance of the three functions we've 
# created so far. To do this, we record the time it takes for one of the
# functions to process a randomly generated list of tuples 10,000 times,
# repeated 50 times for a given combination of function and tuple list.
# We vary the values of `n` and `high`, keeping `low` fixed at 0. 
# Repeating for all three functions gives us a randomly generated sample 
# to estimate mean runtime.
# 
# The reason why we vary `n` and `high` is because indexing a tuple is constant 
# complexity, meaning that changing values of `k`, `max_pos`, and `unique_pos`
# don't have a significant impact on performance. As `n` increases, sorting and 
# traversing become more expensive, and when `high` is greater, 
# there is a greater chance that each tuple has a unique value at `unique_pos`.

# +
def ci_check(sample, cl):
    """
    Checks to ensure that inputs to a confidence interval are valid

    Parameters
    ----------
    sample : 1d np.array or an object coercible as such
        Sample data on which to construct a CI
    cl : float
        Confidence level (must be in [0,1]) that if such an interval were
        constructed many times, CI% of those intervals would capture
        the true population paramater mu
    Raises
    ------
    TypeError
        Raises a TypeError

    Returns
    -------
    None.

    """
    try:
        np.array(sample)
    except:
        raise TypeError("Input cannot be coerced to Numpy Array")
    try:
        np.ndim(np.array(sample)) == 1
    except:
        raise TypeError("Input must be coercible to a 1d Numpy Array")
    try:
        isinstance(cl,float)
    except:
        raise TypeError("Confidence level must be a float")  
    try:
         assert(0 <= cl <= 1)
    except:
        raise ValueError("Confidence level must be in [0,1]")
    return


ci_output = "{est:.2e} [CI: ({lwr:.2e}, {upr:.2e})]"

def std_est_interval(data, cl = 0.95, 
                     ci_format = ci_output):
    """
    Constructs a standard point and interval estimate based on normal theory

    Parameters
    ----------
    data : 1d array-like object coercible to np.array()
        Data passed to the function to create summary statistics
    cl : float, optional
        DESCRIPTION. The default is 0.95.
    ci_format : TYPE, optional
        The format in which the confidence interval is returned.
        If "None" is passed as an argument, then the function returns
        a dictionary of the interval

    Returns
    -------
    Dict, Str
        If ci_format is None, returns dict of interval parameters
        Otherwise, returns formatted string of the CI

    """
    ci_check(data,cl)
    u = np.mean(data)
    alpha = (1 - cl)/2 # CIs are always two-tailed
    se = stats.sem(data)
    z = stats.norm.ppf(1-alpha)
    
    interval = {
            "est" : u,
            "lwr" : u - z*se,
            "upr" : u + z*se,
            "level": cl
            }
    if(ci_format != None):
        return ci_format.format_map(interval) 
    else:
        return interval

def time_func(func, num, terms, l, h, mp, up):
    """
    Parameters
    ----------
    func : function
        function to time
    num : int
        number of tuples
    terms : int
        number of terms in each tuple
    l : int
        lower bound of range to generate test_set
    h : int
        upper bound of range to generate test_set
    mp : int
        position for which we seek maximal value
    up : int
        position for which we seek unique values

    Returns
    -------
    np array
        array of runtimes for the 50 times that func is called 10k times

    """
    test_set = gen_rand_tuples(num, terms, l, h)
    args = [test_set, up, mp]    

    return timeit.repeat(str(func(*args)), 
                         repeat = 50, 
                         number = 10000, 
                         globals = globals())

def time_all_func(function_list):
    """

    Parameters
    ----------
    function_list : list of functions
        functions to test runtime

    Returns
    -------
    time_dict : 3-level nested dictionary
         for each funciton, measures the runtime for various input size i
         and various upper bounds h on the range for which random ints
         are generated to populate the tuples
    """
    time_dict = dict()
    for f in function_list:
        time_dict[f.__name__] = dict()
        for h in range(25,125, 25):
            time_dict[f.__name__][h] = dict()
            for i in range(5,50,5):
                time_dict[f.__name__][h][i] = std_est_interval(
                        time_func(f,i, 10, 0, h, 0, 2))
    return time_dict


f_list = [unique_val_max_pos, dict_uvmp, improved_uvmp]
pretty_f_names = ["Unique Value, Max Position",
                  "Dictionary UVMP",
                  "Improved UVMP"]

table_data = time_all_func(f_list)

count = 0
for f in f_list:
    table_df = pd.DataFrame(table_data[f.__name__])
    table_df = table_df.rename_axis(index = "n")
    table_df = table_df.rename_axis("High", axis = "columns")
    table_df.style.set_caption(pretty_f_names[count] + ", 95% CI")
    #display(HTML(table_df.to_html()))
    display(HTML(table_df.style.set_caption(pretty_f_names[count] + 
                          ", 95% CI").to_html()))
    count += 1
# -

# Both the dictionary and improved UVMP methods show improved performance
# over the original implementation. As expected, increasing 'high' had a
# more pronounced impact on the performance of the dictionary method, but 
# in general the difference in runtime was negligible.