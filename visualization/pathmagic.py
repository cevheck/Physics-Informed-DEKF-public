"""
Fix pathing at start of file
"""

import os, sys

def magic():
    ## pathing
    VisualizationFolder = os.path.dirname(os.path.abspath(__file__))
    PlotFolder = os.path.join(VisualizationFolder, "plots")
    ProjectFolder = os.path.dirname(VisualizationFolder)
    srcFolder = os.path.join(ProjectFolder, "src")
    ResultFolder = os.path.join(ProjectFolder, "results")
    DataFolder = os.path.join(ProjectFolder, "data")

    ## local imports
    sys.path.append(VisualizationFolder)
    sys.path.append(srcFolder)

    ## HMT import
    QuasimoFolder = os.path.dirname(ProjectFolder)
    ResearchFolder = os.path.dirname(QuasimoFolder)
    HAIEMFolder = os.path.join(os.path.join(ResearchFolder, "HMT"), "HMT")
    if os.path.exists(HAIEMFolder) == False:
        HAIEMFolder = os.path.join(ResearchFolder, "HMT_projects/HMT_2.1")
    if not os.path.exists(HAIEMFolder): raise Exception("Provide your path to the HMT here")

    HMT_loc = os.path.join(HAIEMFolder, "src")
    Camfollower_loc = os.path.join(HAIEMFolder, "CamFollower/CamFollowerSimulation")
    sys.path.append(HAIEMFolder)    
    sys.path.append(Camfollower_loc)
    sys.path.append(HMT_loc)

    return PlotFolder, ProjectFolder, DataFolder, ResultFolder