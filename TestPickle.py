import dill
with open("../cs4700-staff-repo/a1-search/student_functions.pkl", "rb") as f:
    c = dill.load(f)

print(sorted([x for x,_ in c]))

for k,v in c:
    globals()[k] = v
print(get_initial_state())