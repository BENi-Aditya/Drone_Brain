from roboflow import Roboflow
rf = Roboflow(api_key="hjCFUfsJBCSxi8KzP8yC")
project = rf.workspace().project("vegetation-gvb0s")
model = project.version(7).model
print(model)