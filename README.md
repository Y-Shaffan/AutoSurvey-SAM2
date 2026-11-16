# AutoSurvey-SAM2

SAM2 implementation for surveying submarine cables.

Below are files created for the task:

* test1.py: Most stable prediction model. Currently, input can only be frames and np arrays denoting x,y co-ordinates for frames. Outputs png mask files, and overlay of said masks over video.
* test0.py: Initial prediction model. Input is largely the same as test1.py, but has no method of streaming input. Inputting large files return memory allocation errors.
* cood_select.py: For manually plotting x,y co-ordinates for frames.
* mask_check.py: For reading the array value of masks outputted by test1.py.
