---
title: Errors and Exceptions
teaching: 30
exercises: 30
---

::::::::::::::::::::::::::::::::::::::: objectives

- To be able to read a traceback, and determine where the error took place and what type of error we are dealing with.
- Be aware of different types of errors (e.g. indentation errors and name errors).
- Understand the process of debugging code containing an error systematically.
- Identify ways of making code less error-prone and more easily tested.

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: questions

- How does Python report errors?
- How can I handle errors in Python programs?
- How can I debug my program?

::::::::::::::::::::::::::::::::::::::::::::::::::

Every programmer encounters errors,
both those who are just beginning,
and those who have been programming for years.
Encountering errors and exceptions can be very frustrating at times,
and can make coding feel like a hopeless endeavour.
However,
understanding what the different types of errors are
and when you are likely to encounter them can help a lot.
Once you know *why* you get certain types of errors,
they become much easier to fix.

Errors in Python have a very specific form,
called a [traceback](../learners/reference.md#traceback).
Let's examine one:

```python
# This code has an intentional error. You can type it directly or
# use it for reference to understand the error message below.
def favorite_ice_cream():
    ice_creams = [
        'chocolate',
        'vanilla',
        'strawberry'
    ]
    print(ice_creams[3])

favorite_ice_cream()
```

```error
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
<ipython-input-1-70bd89baa4df> in <module>()
      9     print(ice_creams[3])
      10
----> 11 favorite_ice_cream()

<ipython-input-1-70bd89baa4df> in favorite_ice_cream()
      7         'strawberry'
      8     ]
----> 9     print(ice_creams[3])
      10
      11 favorite_ice_cream()

IndexError: list index out of range
```

This particular traceback has two levels.
You can determine the number of levels by looking for the number of arrows on the left-hand side.
In this case:

1. The first shows code from the cell above,
  with an arrow pointing to Line 11 (which is `favorite_ice_cream()`).

2. The second shows some code in the function `favorite_ice_cream`,
  with an arrow pointing to Line 9 (which is `print(ice_creams[3])`).

The last level is the actual place where the error occurred.
The other level(s) show what function the program executed to get to the next level down.
So, in this case, the program first performed a
[function call](../learners/reference.md#function-call) to the function `favorite_ice_cream`.
Inside this function,
the program encountered an error on Line 9, when it tried to run the code `print(ice_creams[3])`.

:::::::::::::::::::::::::::::::::::::::::  callout

## Long Tracebacks

Sometimes, you might see a traceback that is very long
\-- sometimes they might even be 20 levels deep!
This can make it seem like something horrible happened,
but the length of the error message does not reflect severity, rather,
it indicates that your program called many functions before it encountered the error.
Most of the time, the actual place where the error occurred is at the bottom-most level,
so you can skip down the traceback to the bottom.


::::::::::::::::::::::::::::::::::::::::::::::::::

So what error did the program actually encounter?
In the last line of the traceback,
Python helpfully tells us the category or type of error (in this case, it is an `IndexError`)
and a more detailed error message (in this case, it says "list index out of range").

If you encounter an error and don't know what it means,
it is still important to read the traceback closely.
That way,
if you fix the error,
but encounter a new one,
you can tell that the error changed.
Additionally,
sometimes knowing *where* the error occurred is enough to fix it,
even if you don't entirely understand the message.

If you do encounter an error you don't recognize,
try looking at the
[official documentation on errors](https://docs.python.org/3/library/exceptions.html).
However,
note that you may not always be able to find the error there,
as it is possible to create custom errors.
In that case,
hopefully the custom error message is informative enough to help you figure out what went wrong.



:::::::::::::::::::::::::::::::::::::::::  callout

## Better errors on newer versions of Python

Newer versions of Python have improved error printouts.  If you are debugging errors, it is often
helpful to use the latest Python version, even if you support older versions of Python.


::::::::::::::::::::::::::::::::::::::::::::::::::

## Syntax Errors

When you forget a colon at the end of a line,
accidentally add one space too many when indenting under an `if` statement,
or forget a parenthesis,
you will encounter a [syntax error](../learners/reference.md#syntax-error).
This means that Python couldn't figure out how to read your program.
This is similar to forgetting punctuation in English:
for example,
this text is difficult to read there is no punctuation there is also no capitalisation
why is this hard because you have to figure out where each sentence ends
you also have to figure out where each sentence begins
to some extent it might be ambiguous if there should be a sentence break or not

People can typically figure out what is meant by text with no punctuation,
but people are much smarter than computers.
If Python doesn't know how to read the program,
it will give up and inform you with an error.
For example:

```python
def some_function()
    msg = 'hello, world!'
    print(msg)
     return msg
```

```error
  File "<ipython-input-3-6bb841ea1423>", line 1
    def some_function()
                       ^
SyntaxError: invalid syntax
```

Here, Python tells us that there is a `SyntaxError` on line 1,
and even puts a little arrow in the place where there is an issue.
In this case the problem is that the function definition is missing a colon at the end.

Actually, the function above has *two* issues with syntax.
If we fix the problem with the colon,
we see that there is *also* an `IndentationError`,
which means that the lines in the function definition do not all have the same indentation:

```python
def some_function():
    msg = 'hello, world!'
    print(msg)
     return msg
```

```error
  File "<ipython-input-4-ae290e7659cb>", line 4
    return msg
    ^
IndentationError: unexpected indent
```

Both `SyntaxError` and `IndentationError` indicate a problem with the syntax of your program,
but an `IndentationError` is more specific:
it *always* means that there is a problem with how your code is indented.

:::::::::::::::::::::::::::::::::::::::::  callout

## Tabs and Spaces

Some indentation errors are harder to spot than others.
In particular, mixing spaces and tabs can be difficult to spot
because they are both [whitespace](../learners/reference.md#whitespace).
In the example below, the first two lines in the body of the function
`some_function` are indented with tabs, while the third line — with spaces.
If you're working in a Jupyter notebook, be sure to copy and paste this example
rather than trying to type it in manually because Jupyter automatically replaces
tabs with spaces.

```python
def some_function():
	msg = 'hello, world!'
	print(msg)
        return msg
```

Visually it is impossible to spot the error.
Fortunately, Python does not allow you to mix tabs and spaces.

```error
  File "<ipython-input-5-653b36fbcd41>", line 4
    return msg
              ^
TabError: inconsistent use of tabs and spaces in indentation
```

::::::::::::::::::::::::::::::::::::::::::::::::::

## Variable Name Errors

Another very common type of error is called a `NameError`,
and occurs when you try to use a variable that does not exist.
For example:

```python
print(a)
```

```error
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-7-9d7b17ad5387> in <module>()
----> 1 print(a)

NameError: name 'a' is not defined
```

Variable name errors come with some of the most informative error messages,
which are usually of the form "name 'the\_variable\_name' is not defined".

Why does this error message occur?
That's a harder question to answer,
because it depends on what your code is supposed to do.
However,
there are a few very common reasons why you might have an undefined variable.
The first is that you meant to use a
[string](../learners/reference.md#string), but forgot to put quotes around it:

```python
print(hello)
```

```error
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-8-9553ee03b645> in <module>()
----> 1 print(hello)

NameError: name 'hello' is not defined
```

The second reason is that you might be trying to use a variable that does not yet exist.
In the following example,
`count` should have been defined (e.g., with `count = 0`) before the for loop:

```python
for number in range(10):
    count = count + number
print('The count is:', count)
```

```error
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-9-dd6a12d7ca5c> in <module>()
      1 for number in range(10):
----> 2     count = count + number
      3 print('The count is:', count)

NameError: name 'count' is not defined
```

Finally, the third possibility is that you made a typo when you were writing your code.
Let's say we fixed the error above by adding the line `Count = 0` before the for loop.
Frustratingly, this actually does not fix the error.
Remember that variables are [case-sensitive](../learners/reference.md#case-sensitive),
so the variable `count` is different from `Count`. We still get the same error,
because we still have not defined `count`:

```python
Count = 0
for number in range(10):
    count = count + number
print('The count is:', count)
```

```error
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-10-d77d40059aea> in <module>()
      1 Count = 0
      2 for number in range(10):
----> 3     count = count + number
      4 print('The count is:', count)

NameError: name 'count' is not defined
```

## Index Errors

Next up are errors having to do with containers (like lists and strings) and the items within them.
If you try to access an item in a list or a string that does not exist,
then you will get an error.
This makes sense:
if you asked someone what day they would like to get coffee,
and they answered "caturday",
you might be a bit annoyed.
Python gets similarly annoyed if you try to ask it for an item that doesn't exist:

```python
letters = ['a', 'b', 'c']
print('Letter #1 is', letters[0])
print('Letter #2 is', letters[1])
print('Letter #3 is', letters[2])
print('Letter #4 is', letters[3])
```

```output
Letter #1 is a
Letter #2 is b
Letter #3 is c
```

```error
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
<ipython-input-11-d817f55b7d6c> in <module>()
      3 print('Letter #2 is', letters[1])
      4 print('Letter #3 is', letters[2])
----> 5 print('Letter #4 is', letters[3])

IndexError: list index out of range
```

Here,
Python is telling us that there is an `IndexError` in our code,
meaning we tried to access a list index that did not exist.

## File Errors

The last type of error we'll cover today
are those associated with reading and writing files: `FileNotFoundError`.
If you try to read a file that does not exist,
you will receive a `FileNotFoundError` telling you so.
If you attempt to write to a file that was opened read-only, Python 3
returns an `UnsupportedOperationError`.
More generally, problems with input and output manifest as
`OSError`s, which may show up as a more specific subclass; you can see
[the list in the Python docs](https://docs.python.org/3/library/exceptions.html#os-exceptions).
They all have a unique UNIX `errno`, which is you can see in the error message.

```python
file_handle = open('myfile.txt', 'r')
```

```error
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
<ipython-input-14-f6e1ac4aee96> in <module>()
----> 1 file_handle = open('myfile.txt', 'r')

FileNotFoundError: [Errno 2] No such file or directory: 'myfile.txt'
```

One reason for receiving this error is that you specified an incorrect path to the file.
For example,
if I am currently in a folder called `myproject`,
and I have a file in `myproject/writing/myfile.txt`,
but I try to open `myfile.txt`,
this will fail.
The correct path would be `writing/myfile.txt`.
It is also possible that the file name or its path contains a typo.

A related issue can occur if you use the "read" flag instead of the "write" flag.
Python will not give you an error if you try to open a file for writing
when the file does not exist.
However,
if you meant to open a file for reading,
but accidentally opened it for writing,
and then try to read from it,
you will get an `UnsupportedOperation` error
telling you that the file was not opened for reading:

```python
file_handle = open('myfile.txt', 'w')
file_handle.read()
```

```error
---------------------------------------------------------------------------
UnsupportedOperation                      Traceback (most recent call last)
<ipython-input-15-b846479bc61f> in <module>()
      1 file_handle = open('myfile.txt', 'w')
----> 2 file_handle.read()

UnsupportedOperation: not readable
```

These are the most common errors with files,
though many others exist.
If you get an error that you've never seen before,
searching the Internet for that error type
often reveals common reasons why you might get that error.



Once testing has uncovered problems,
the next step is to fix them.
Many novices do this by making more-or-less random changes to their code
until it seems to produce the right answer,
but that's very inefficient
(and the result is usually only correct for the one case they're testing).
The more experienced a programmer is,
the more systematically they debug,
and most follow some variation on the rules explained below.

## Know What It's Supposed to Do

The first step in debugging something is to
*know what it's supposed to do*.
"My program doesn't work" isn't good enough:
in order to diagnose and fix problems,
we need to be able to tell correct output from incorrect.
If we can write a test case for the failing case --- i.e.,
if we can assert that with *these* inputs,
the function should produce *that* result ---
then we're ready to start debugging.
If we can't,
then we need to figure out how we're going to know when we've fixed things.

But writing test cases for scientific software is frequently harder than
writing test cases for commercial applications,
because if we knew what the output of the scientific code was supposed to be,
we wouldn't be running the software:
we'd be writing up our results and moving on to the next program.
In practice,
scientists tend to do the following:

1. *Test with simplified data.*
  Before doing statistics on a real data set,
  we should try calculating statistics for a single record,
  for two identical records,
  for two records whose values are one step apart,
  or for some other case where we can calculate the right answer by hand.

2. *Test a simplified case.*
  If our program is supposed to simulate
  magnetic eddies in rapidly-rotating blobs of supercooled helium,
  our first test should be a blob of helium that isn't rotating,
  and isn't being subjected to any external electromagnetic fields.
  Similarly,
  if we're looking at the effects of climate change on speciation,
  our first test should hold temperature, precipitation, and other factors constant.

3. *Compare to an oracle.*
  A [test oracle](../learners/reference.md#test-oracle)
  is something whose results are trusted,
  such as experimental data, an older program, or a human expert.
  We use test oracles to determine if our new program produces the correct results.
  If we have a test oracle,
  we should store its output for particular cases
  so that we can compare it with our new results as often as we like
  without re-running that program.

4. *Check conservation laws.*
  Mass, energy, and other quantities are conserved in physical systems,
  so they should be in programs as well.
  Similarly,
  if we are analyzing patient data,
  the number of records should either stay the same or decrease
  as we move from one analysis to the next
  (since we might throw away outliers or records with missing values).
  If "new" patients start appearing out of nowhere as we move through our pipeline,
  it's probably a sign that something is wrong.

5. *Visualise.*
  Data analysts frequently use simple visualisations to check both
  the science they're doing
  and the correctness of their code
  (just as we did in the [opening lesson](02-numpy.html) of this tutorial).
  This should not be the only debugging method you rely on, since visual comparisons are hard to automate.

## Make It Fail Every Time

We can only debug something when it fails,
so the second step is always to find a test case that
*makes it fail every time*.
The "every time" part is important because
few things are more frustrating than debugging an intermittent problem:
if we have to call a function a dozen times to get a single failure,
the odds are good that we'll scroll past the failure when it actually occurs.

As part of this,
it's always important to check that our code is "plugged in",
i.e.,
that we're actually exercising the problem that we think we are.
Every programmer has spent hours chasing a bug,
only to realize that they were actually calling their code on the wrong data set
or with the wrong configuration parameters,
or are using the wrong version of the software entirely.
Mistakes like these are particularly likely to happen when we're tired,
frustrated,
and up against a deadline,
which is one of the reasons late-night (or overnight) coding sessions
are almost never worthwhile.

## Make It Fail Fast

If it takes 20 minutes for the bug to surface,
we can only do three experiments an hour.
This means that we'll get less data in more time and that
we're more likely to be distracted by other things as we wait for our program to fail,
which means the time we *are* spending on the problem is less focused.
It's therefore critical to *make it fail fast*.

As well as making the program fail fast in time,
we want to make it fail fast in space,
i.e.,
we want to localize the failure to the smallest possible region of code:

1. The smaller the gap between cause and effect,
  the easier the connection is to find.
  Many programmers therefore use a divide and conquer strategy to find bugs,
  i.e.,
  if the output of a function is wrong,
  they check whether things are OK in the middle,
  then concentrate on either the first or second half,
  and so on.

2. N things can interact in N! different ways,
  so every line of code that *isn't* run as part of a test
  means more than one thing we don't need to worry about.

## Change One Thing at a Time, For a Reason

Replacing random chunks of code is unlikely to do much good.
(After all,
if you got it wrong the first time,
you'll probably get it wrong the second and third as well.)
Good programmers therefore
*change one thing at a time, for a reason*.
They are either trying to gather more information
("is the bug still there if we change the order of the loops?")
or test a fix
("can we make the bug go away by sorting our data before processing it?").

Every time we make a change,
however small,
we should re-run our tests immediately,
because the more things we change at once,
the harder it is to know what's responsible for what
(those N! interactions again).
And we should re-run *all* of our tests:
more than half of fixes made to code introduce (or re-introduce) bugs,
so re-running all of our tests tells us whether we have regressed.

## Keep Track of What You've Done

Good scientists keep track of what they've done
so that they can reproduce their work,
and so that they don't waste time repeating the same experiments
or running ones whose results won't be interesting.
Similarly,
debugging works best when we
*keep track of what we've done*
and how well it worked.
If we find ourselves asking,
"Did left followed by right with an odd number of lines cause the crash?
Or was it right followed by left?
Or was I using an even number of lines?"
then it's time to step away from the computer,
take a deep breath,
and start working more systematically.

Records are particularly useful when the time comes to ask for help.
People are more likely to listen to us
when we can explain clearly what we did,
and we're better able to give them the information they need to be useful.

:::::::::::::::::::::::::::::::::::::::::  callout

## Version Control Revisited

Version control is often used to reset software to a known state during debugging,
and to explore recent changes to code that might be responsible for bugs.
In particular,
most version control systems (e.g. Git, Mercurial) have:

1. a `blame` command that shows who last changed each line of a file;
2. a `bisect` command that helps with finding the commit that introduced an
  issue.
  

::::::::::::::::::::::::::::::::::::::::::::::::::

## Be Humble

And speaking of help:
if we can't find a bug in a reasonable amount of time,
we should *be humble* and [ask for help](./01-intro.md#getting-help).
Explaining the problem to someone else is often useful,
since hearing what we're thinking helps us spot inconsistencies and hidden assumptions.
If you don't have someone nearby to share your problem description with, get a
[rubber duck](https://en.wikipedia.org/wiki/Rubber_duck_debugging)!

Asking for help also helps alleviate confirmation bias.
If we have just spent an hour writing a complicated program,
we want it to work,
so we're likely to keep telling ourselves why it should,
rather than searching for the reason it doesn't.
People who aren't emotionally invested in the code can be more objective,
which is why they're often able to spot the simple mistakes we have overlooked.

Part of being humble is learning from our mistakes.
Programmers tend to get the same things wrong over and over:
either they don't understand the language and libraries they're working with,
or their model of how things work is wrong.
In either case,
taking note of why the error occurred
and checking for it next time
quickly turns into not making the mistake at all.

And that is what makes us most productive in the long run.
As the saying goes,
*A week of hard work can sometimes save you an hour of thought*.
If we train ourselves to avoid making some kinds of mistakes,
to break our code into modular, testable chunks,
and to turn every assumption (or mistake) into an assertion,
it will actually take us *less* time to produce working programs,
not more.

:::::::::::::::::::::::::::::::::::::::  challenge

## Reading Error Messages

Read the Python code and the resulting traceback below, and answer the following questions:

1. How many levels does the traceback have?
2. What is the function name where the error occurred?
3. On which line number in this function did the error occur?
4. What is the type of error?
5. What is the error message?

```python
# This code has an intentional error. Do not type it directly;
# use it for reference to understand the error message below.
def print_message(day):
    messages = [
        'Hello, world!',
        'Today is Tuesday!',
        'It is the middle of the week.',
        'Today is Donnerstag in German!',
        'Last day of the week!',
        'Hooray for the weekend!',
        'Aw, the weekend is almost over.'
    ]
    print(messages[day])

def print_sunday_message():
    print_message(7)

print_sunday_message()
```

```error
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
<ipython-input-7-3ad455d81842> in <module>
     16     print_message(7)
     17 
---> 18 print_sunday_message()
     19 

<ipython-input-7-3ad455d81842> in print_sunday_message()
     14 
     15 def print_sunday_message():
---> 16     print_message(7)
     17 
     18 print_sunday_message()

<ipython-input-7-3ad455d81842> in print_message(day)
     11         'Aw, the weekend is almost over.'
     12     ]
---> 13     print(messages[day])
     14 
     15 def print_sunday_message():

IndexError: list index out of range
```

:::::::::::::::  solution

## Solution

1. 3 levels
2. `print_message`
3. 13
4. `IndexError`
5. `list index out of range`. You can then infer that
  `7` is not the right index to use with `messages`.
  
  

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::  challenge

## Debug With a Neighbor

Take a function that you have written today, and introduce a tricky bug.
Your function should still run, but will give the wrong output.
Switch seats with your neighbor and attempt to debug
the bug that they introduced into their function.
Which of the principles discussed above did you find helpful?


::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::  challenge

## Not Supposed to be the Same

You are assisting a researcher with Python code that computes the
Body Mass Index (BMI) of patients.  The researcher is concerned because
all patients seemingly have unusual and identical BMIs, despite having different
physiques.  BMI is calculated as **weight in kilograms**
divided by the square of **height in metres**.

Use the debugging principles in this exercise and locate problems
with the code. What suggestions would you give the researcher for
ensuring any later changes they make work correctly? What bugs do you spot?

```python
patients = [[70, 1.8], [80, 1.9], [150, 1.7]]

def calculate_bmi(weight, height):
    return weight / (height ** 2)

for patient in patients:
    weight, height = patients[0]
    bmi = calculate_bmi(height, weight)
    print("Patient's BMI is:", bmi)
```

```output
Patient's BMI is: 0.000367
Patient's BMI is: 0.000367
Patient's BMI is: 0.000367
```

:::::::::::::::  solution

## Solution

### Suggestions for debugging

- Add printing statement in the `calculate_bmi` function, like `print('weight:', weight, 'height:', height)`, to make clear that what the BMI is based on.
- Change `print("Patient's BMI is: %f" % bmi)` to `print("Patient's BMI (weight: %f, height: %f) is: %f" % (weight, height, bmi))`, in order to be able to distinguish bugs in the function from bugs in the loop.

### Bugs found

- The loop is not being utilised correctly. `height` and `weight` are always
  set as the first patient's data during each iteration of the loop.

- The height/weight variables are reversed in the function call to
  `calculate_bmi(...)`, the correct BMIs are 21.604938, 22.160665 and 51.903114.
  
  

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: keypoints

- Tracebacks can look intimidating, but they give us a lot of useful information about what went wrong in our program, including where the error occurred and what type of error it was.
- An error having to do with the 'grammar' or syntax of the program is called a `SyntaxError`. If the issue has to do with how the code is indented, then it will be called an `IndentationError`.
- A `NameError` will occur when trying to use a variable that does not exist. P
- Containers like lists and strings will generate errors if you try to access items in them that do not exist. This type of error is called an `IndexError`.
- Trying to read a file that does not exist will give you an `FileNotFoundError`. Trying to read a file that is open for writing, or writing to a file that is open for reading, will give you an `IOError`.
- Know what code is supposed to do *before* trying to debug it.
- Make it fail every time.
- Make it fail fast.
- Change ONLY one thing at a time, and for a reason.
- Keep track of what you've done.
- Be humble and patient.
- Use help.

::::::::::::::::::::::::::::::::::::::::::::::::::


