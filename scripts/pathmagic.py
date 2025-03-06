"""
Fix pathing at start of file
"""

import os, sys

def magic():
    ## pathing
    ScriptFolder = os.path.dirname(os.path.abspath(__file__))
    ProjectFolder = os.path.dirname(ScriptFolder)
    srcFolder = os.path.join(ProjectFolder, "src")
    ResultFolder = os.path.join(ProjectFolder, "results")
    DataFolder = os.path.join(ProjectFolder, "data")

    ## local imports
    sys.path.append(ScriptFolder)
    sys.path.append(srcFolder)

    ## HMT import
    ProjectQfolder = os.path.dirname(ProjectFolder)       # my custom pathing
    ResearchFolder = os.path.dirname(ProjectQfolder)      # my custom pathing
    ProjectHMTfolder = os.path.join(ResearchFolder, "HMT")
    if os.path.exists(ProjectHMTfolder) == False:
        ProjectHMTfolder = os.path.join(ResearchFolder, "HMT_projects/HMT_2.1")
    if not os.path.exists(ProjectHMTfolder): raise Exception("Provide your path to the HMT here. Without acces rights to the HMT, the (hybrid) models will not work")

    HMT_loc = os.path.join(ProjectHMTfolder, "HMT/src")
    Camfollower_loc = os.path.join(ProjectHMTfolder, "CamFollower/CamFollowerSimulation")
    sys.path.append(ProjectHMTfolder)    
    sys.path.append(Camfollower_loc)
    sys.path.append(HMT_loc)

    return ProjectFolder, DataFolder, ResultFolder