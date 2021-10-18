import settings_ash
import os
import numpy as np
import time
from functools import wraps
import math

from ash import ashpath

# ANSI colors: http://jafrog.com/2013/11/23/colors-in-terminal.html
if settings_ash.settings_dict["use_ANSI_color"] is True:
    class BC:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        OKMAGENTA = '\033[95m'
        OKRED = '\033[31m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        END = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
else:
    class BC:
        HEADER = ''
        OKBLUE = ''
        OKGREEN = ''
        OKMAGENTA = ''
        OKRED = ''
        WARNING = ''
        FAIL = ''
        END = ''
        BOLD = ''
        UNDERLINE = ''

# Julia load interface
# TODO: Avoid reloading
julia_loaded = False


def load_julia_interface():
    global julia_loaded
    global JuliaMain
    if julia_loaded is False:
        # print("julia loaded false")
        # from julia.api import Julia
        # jl = Julia(compiled_modules=False)
        from julia import Main as JuliaMain
        JuliaMain.include(ashpath + "/functions/functions_julia.jl")
        julia_loaded = True
    # else:
    #    print("Julia loaded true")
    # print("JuliaMain:", JuliaMain)
    # print("JuliaMain include:", JuliaMain.include)
    # print(JuliaMain.__dict__)
    return JuliaMain.Juliafunctions


# Get ranges of integers from list. Returns string of ranges. Used to communitcate with crest and xtb
# example: input: [1,2,3,4,5,6,20,21,22,23,500,700,701,702,1000,1100,1101]
# output: '1-6,20-23,500,700-702,1000,1100-1101'
def int_ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    l_of_tuples = list(zip(edges, edges))

    newstring = ""
    for i in l_of_tuples:
        if i[0] != i[1]:
            newstring += str(i[0]) + '-' + str(i[1]) + ','
        else:
            newstring += str(i[0]) + ','
    # remove final ,
    newstring = newstring[0:-1]
    return newstring


# Debug print. Behaves like print but reads global debug var first
def printdebug(string, var=''):
    if settings_ash.settings_dict["debugflag"] is True:
        print(BC.OKRED, string, var, BC.END)


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn:" + fn.__name__ + " took " + str(t2 - t1) + " seconds")
        return result

    return measure_time


# Grep-style function to find a line in file and return a list of words
# TODO: Make more advanced
def pygrep(string, file):
    with open(file) as f:
        for line in f:
            if string in line:
                stringlist = line.split()
                return stringlist


# Multiple match version. Replace pygrep ?
def pygrep2(string, file):
    l = []
    with open(file) as f:
        for line in f:
            if string in line:
                l.append(string)

    return l


# Give difference of two lists, sorted. List1: Bigger list
def listdiff(list1, list2):
    diff = (list(set(list1) - set(list2)))
    diff.sort()
    return diff


# Range function for floats
# Using round to deal with floating-point problem : 0.6+0.3 =0.89999
def frange(start, stop=None, step=None, rounddigits=4):
    # if stop and step argument is None set start=0.0 and step = 1.0
    start = float(start)
    if stop is None:
        stop = start + 0.0
        start = 0.0
    if step is None:
        step = 1.0
    # print("start= ", start, "stop= ", stop, "step= ", step)
    count = 0
    while True:
        temp = round(float(start + count * step), rounddigits)
        # print("temp:", temp)
        if step > 0 and temp >= stop:
            break
        elif step < 0 and temp <= stop:
            break
        yield temp
        count += 1


# FUNCTIONS TO PRINT MODULE AND SUBMODULE HEADERS

# mainmodule header
def print_line_with_mainheader(line):
    length = len(line)
    offset = 12
    outer_line = f"{BC.OKGREEN}{'#' * (length + offset)}{BC.END}"
    midline = f"{BC.OKGREEN}#{' ' * (length + offset - 2)}#{BC.END}"
    inner_line = f"{BC.OKGREEN}#{' ' * (offset//2 - 1)}{BC.BOLD}{line}{' ' * (offset//2 - 1)}#{BC.END}"
    print("\n")
    print(outer_line.center(80))
    print(midline.center(80))
    print(inner_line.center(80))
    print(midline.center(80))
    print(outer_line.center(80))
    print("\n")


# Submodule header
def print_line_with_subheader1(line):
    print("")
    print(f"{BC.OKBLUE}{'-' * 80}{BC.END}")
    print(f"{BC.OKBLUE}{BC.BOLD}{line.center(80)}{BC.END}")
    print(f"{BC.OKBLUE}{'-' * 80}{BC.END}")
    print("")


# Smaller header
def print_line_with_subheader2(line):
    print("")
    length = len(line)
    print(f"{BC.OKBLUE}{'-' * length}{BC.END}")
    print(f"{BC.OKBLUE}{BC.BOLD}{line}{BC.END}")
    print(f"{BC.OKBLUE}{'-' * length}{BC.END}")


# Inserts line into file for matched string.
# option: Once=True means only added for first match
def insert_line_into_file(file, string, addedstring, Once=True):
    Added = False
    with open(file, 'r') as ffr:
        contents = ffr.readlines()
    with open(file, 'w') as ffw:
        for l in contents:
            ffw.write(l)
            if string in l:
                if Added is False:
                    ffw.write(addedstring + '\n')
                    if Once is True:
                        Added = True


def blankline():
    print("")


# Can variable be converted into integer
def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


# Is integer odd
def isodd(n):
    if (n % 2) == 0:
        return False
    else:
        return True


# Compare sign of two numbers. Return True if same sign, return False if opposite sign
def is_same_sign(a, b):
    if a * b < 0:
        return False
    elif a * b > 0:
        return True


# Is it possible to interpret string/number as float.
# Note: integer variable/string can be interpreted. See also is_string_float_withdecimal below
def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Is string a float with single decimal point
def is_string_float_withdecimal(s):
    # Checking if single . in string
    if s.count('.') != 1:
        return False
    # Check if number can be interpreted as float
    try:
        float(s)
        return True
    except ValueError:
        return False


# Check if list of integers is sorted or not.
def is_integerlist_ordered(list):
    list_s = sorted(list)
    if list == list_s:
        return True
    else:
        return False


def islist(l):
    if type(l) == list:
        return True
    else:
        return False


# Read lines of file by slurping.
# def readlinesfile(filename):
#     try:
#         f = open(filename)
#         out = f.readlines()
#         f.close()
#     except IOError:
#         print('File %s does not exist!' % (filename))
#         exit(12)
#     return out


# Find substring of string between left and right parts
def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]


# Read list of integers from file. Output list of integers. Ignores blanklines, return chars, non-int characters
# offset option: shifts integers by a value (e.g. 1 or -1)
def read_intlist_from_file(filename, offset=0):
    intlist = []
    try:
        with open(filename, "r") as f:
            for line in f:
                for l in line.split():
                    # Removing non-numeric part
                    l = ''.join(i for i in l if i.isdigit())
                    if isint(l):
                        intlist.append(int(l) + offset)
    except FileNotFoundError:
        print(f"File '{filename}' does not exists!")
        exit()
    intlist.sort()
    return intlist


# Read list of flaots from file. Output list of floats.
# Works for single-line with numbers and multi-lines
def read_floatlist_from_file(filename):
    floatlist = []
    try:
        with open(filename, "r") as f:
            for line in f:
                for l in line.split():
                    if isfloat(l):
                        floatlist.append(float(l))
    except FileNotFoundError:
        print(f"File '{filename}' does not exists!")
        exit()
    floatlist.sort()
    return floatlist


# Write a string to file simply
def writestringtofile(string, file):
    with open(file, 'w') as f:
        f.write(string)


# Write a Python list to file simply
def writelisttofile(pylist, file):
    with open(file, 'w') as f:
        for l in pylist:
            f.write(str(l) + ' ')
    print("Wrote list to file:", file)


# Natural (human) sorting of list
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# Reverse read function.
def reverse_lines(filename, BUFSIZE=20480):
    # f = open(filename, "r")
    filename.seek(0, 2)
    p = filename.tell()
    remainder = ""
    while True:
        sz = min(BUFSIZE, p)
        p -= sz
        filename.seek(p)
        buf = filename.read(sz) + remainder
        if '\n' not in buf:
            remainder = buf
        else:
            i = buf.index('\n')
            for L in buf[i + 1:].split("\n")[::-1]:
                yield L
            remainder = buf[:i]
        if p == 0:
            break
    yield remainder


def clean_number(number):
    return np.real_if_close(number)


# Function to get unique values
def uniq(seq, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


# Extract column from matrix
def column(matrix, i):
    return [row[i] for row in matrix]


# Various function to print time of module/step. Will add time also to Timings object

def print_time_rel(timestamp, modulename='Unknown', moduleindex=4):
    secs = time.time() - timestamp
    mins = secs / 60
    print_line_with_subheader2(
        "Time to calculate step ({}): {:3.1f} seconds, {:3.1f} minutes.".format(modulename, secs, mins))
    # print("-------------------------------------------------------------------")
    # print("Time to calculate step ({}): {:3.1f} seconds, {:3.1f} minutes.".format(modulename, secs, mins))
    # print("-------------------------------------------------------------------")
    # Adding time to Timings object
    timingsobject.add(modulename, secs, moduleindex=moduleindex)


def print_time_rel_and_tot(timestampA, timestampB, modulename='Unknown', moduleindex=4):
    secsA = time.time() - timestampA
    minsA = secsA / 60
    # hoursA=minsA/60
    secsB = time.time() - timestampB
    minsB = secsB / 60
    # hoursB=minsB/60
    print("-------------------------------------------------------------------")
    print("Time to calculate step ({}): {:3.1f} seconds, {:3.1f} minutes.".format(modulename, secsA, minsA))
    print("Total Walltime: {:3.1f} seconds, {:3.1f} minutes.".format(secsB, minsB))
    print("-------------------------------------------------------------------")
    # Adding time to Timings object
    timingsobject.add(modulename, secsA, moduleindex=moduleindex)


def print_time_tot_color(time_initial, modulename='Unknown', moduleindex=4):
    # hoursA=minsA/60
    secs = time.time() - time_initial
    mins = secs / 60
    # hoursB=minsB/60
    print(BC.WARNING, "-------------------------------------------------------------------", BC.END)
    print(BC.WARNING, "ASH Total Walltime: {:3.1f} seconds, {:3.1f} minutes.".format(secs, mins), BC.END)
    print(BC.WARNING, "-------------------------------------------------------------------", BC.END)
    # Adding time to Timings object
    timingsobject.add(modulename, secs, moduleindex=moduleindex)


# Keep track of module runtimes
class Timings:
    def __init__(self):
        self.simple_dict = {}
        self.module_count = {}
        self.module_indices = {}
        self.totalsumtime = 0

    def add(self, modulename, mtime, moduleindex=4):

        # Adding time to dictionary
        if modulename in self.simple_dict:
            self.simple_dict[modulename] += mtime
        else:
            self.simple_dict[modulename] = mtime

        # Adding moduleindex to dictionary
        if modulename not in self.module_indices:
            self.module_indices[modulename] = moduleindex

        # Adding times called
        if modulename in self.module_count:
            self.module_count[modulename] += 1
        else:
            self.module_count[modulename] = 1

        self.totalsumtime += mtime

    # Distinguish and sort between: 
    # workflows (thermochem_protol, PES, calc_surface etc.): 0
    # jobtype (optimizer,Singlepoint,Anfreq,Numfreq): 1
    # theory-run (ORCAtheory run, QM/MM run, MM run etc.): 2 
    # various: 3 
    # others (calc connectivity etc.): 4

    def print(self, inittime):
        totalwalltime = time.time() - inittime

        print("Note: module-timings are a work in progress (report bugs!)")
        print("To turn off timing output add to settings file: ~/ash_user_settings.ini")
        print("print_full_timings = False   ")
        ######################
        # Old way of printing
        # Sort dict by value
        # simple_dict_ordered=dict(sorted(self.simple_dict.items(), key=lambda item: item[1]))
        # print("")
        # idea: if module labelled submodule add ---> or something and dont't include in totalsumtime above ??
        # print("{:35}{:>20}{:>20}{:>17}".format("Modulename", "Time (sec)", "Percentage of total", "Times called"))
        # print("-"*100)
        # for dictitem in simple_dict_ordered:
        #    mmtime=simple_dict_ordered[dictitem]
        #    time_per= 100*(mmtime/totalwalltime)
        #    print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
        # print("")
        # print("{:35}{:>20.2f}".format("Sum of all moduletimes (flawed)", self.totalsumtime))
        # print("{:35}{:>20.2f}{:>10}".format("Total walltime", totalwalltime, 100.0))
        # print("-"*100)

        print("")
        print("{:35}{:>20}{:>20}{:>17}".format("Modulename", "Time (sec)", "Percentage of total", "Times called"))
        print("-" * 100)

        # Lists of dictitems by module_labels
        # Workflows: thermochemprotocol, calc_surface, benchmarking etc.
        dictitems_index0 = [i for i in self.simple_dict if self.module_indices[i] == 0]
        # Jobtype: Singlepoint, Opt, freq
        dictitems_index1 = [i for i in self.simple_dict if self.module_indices[i] == 1]
        # Theory run: ORCATHeory, QM/MM Theory etc
        dictitems_index2 = [i for i in self.simple_dict if self.module_indices[i] == 2]
        # NOTE: currently not using index 3. Disabled until a good reason for it
        # dictitems_index3=[i for i in self.simple_dict if self.module_indices[i] == 3]
        # Other small modules. 4 is default
        dictitems_index4 = [i for i in self.simple_dict if self.module_indices[i] == 4]

        if len(dictitems_index0) != 0:
            print("Workflow modules")
            print("-" * 30)
            for dictitem in dictitems_index0:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        if len(dictitems_index1) != 0:
            print("Jobtype modules")
            print("-" * 30)
            for dictitem in dictitems_index1:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        if len(dictitems_index2) != 0:
            print("Theory-run modules")
            print("-" * 30)
            for dictitem in dictitems_index2:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        # if len(dictitems_index3) !=0 :
        # print("Various modules")
        # print("-"*30)
        # for dictitem in dictitems_index3:
        #    mmtime=self.simple_dict[dictitem]
        #    time_per= 100*(mmtime/totalwalltime)
        #    print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
        # print("")
        if len(dictitems_index4) != 0:
            print("Other modules")
            print("-" * 30)
            for dictitem in dictitems_index4:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        print("")
        print("{:35}{:>20.2f}".format("Sum of all moduletimes (flawed)", self.totalsumtime))
        print("{:35}{:>20.2f}{:>10}".format("Total walltime", totalwalltime, 100.0))


# Creating object
timingsobject = Timings()
