toggle := 0  ; Initialize toggle variable to 0 (false/off)

; Define a hotkey, for example, F1 to toggle the loop on and off
P::  ; When P is pressed
toggle := !toggle  ; Invert the value of toggle (0 becomes 1, 1 becomes 0)
if (toggle)  ; If toggle is now true (1)
{
    SetTimer, PressKeys, 0  ; Start the timer immediately
}
else  ; If toggle is now false (0)
{
    SetTimer, PressKeys, Off  ; Turn off the timer
}
return

PressKeys:
Send, {F2}  ; Press the F2 key
Sleep, 1000  ; Wait for 1 second (1000 milliseconds)
Send, {F6}  ; Press the F6 key
Sleep, 3500  ; Wait for 4 seconds (4000 milliseconds)
return