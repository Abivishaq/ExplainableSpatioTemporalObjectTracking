from helpers.encoders import time_external

for i in [0,33,69,118]:
    time_i = 380+i*10
    time_sem = time_external(time_i)
    hrs = time_sem[2]
    mins = time_sem[3]

    print(f"Time: {i} : {time_i}-> Hours: {hrs}, Minutes: {mins}")