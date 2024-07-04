import numpy as np
import vtk
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params import *

# Color map for different classes (you can customize this list with different colors)
color_map = get_colors(n_class)

class VtkPointCloud:
    def __init__(self, zMin=0, zMax=0, maxNumPoints=1e6, pointSize=2):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        self.vtkActor.GetProperty().SetPointSize(pointSize)
        if zMin == 0 and zMax == 0:
            mapper.SetScalarVisibility(0)
        else:
            mapper.SetScalarVisibility(1)

    def addPoint(self, point, color):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
            self.vtkColors.InsertNextTuple3(color[0], color[1], color[2])
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()
        self.vtkColors.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkColors = vtk.vtkUnsignedCharArray()
        self.vtkColors.SetNumberOfComponents(3)
        self.vtkColors.SetName('Colors')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')
        self.vtkPolyData.GetPointData().SetScalars(self.vtkColors)

# Load point clouds
all_lines = []
point_clouds = []

for i in range(n_class):
    if os.path.exists(pointcloud_filenames[i]):
        lines = np.loadtxt(pointcloud_filenames[i], delimiter=',', unpack=False)
        if lines.size > 0:
            all_lines.append(lines)
            point_clouds.append(lines)
        else:
            point_clouds.append(None)
    else:
        point_clouds.append(None)

# Compute the bounding box of the point clouds
if all_lines:
    all_lines_combined = np.concatenate(all_lines, axis=0)
    min_vals = np.min(all_lines_combined, axis=0)
    max_vals = np.max(all_lines_combined, axis=0)
    axis_length = max((max_vals[0] - min_vals[0]) * 1.5, (max_vals[1] - min_vals[1]) * 1.5, (max_vals[2] - min_vals[2]) * 1.5)
else:
    min_vals = np.array([0, 0, 0])
    max_vals = np.array([1, 1, 1])
    axis_length = 1.0

# Create VTK point clouds with different colors
vtkPointClouds = []
for i in range(n_class):
    if point_clouds[i] is not None:
        vtkPointCloud = VtkPointCloud(zMin=min_vals[2], zMax=max_vals[2])
        for point in point_clouds[i]:
            vtkPointCloud.addPoint(point, color_map[i % len(color_map)])
        vtkPointClouds.append(vtkPointCloud)

# Renderer
renderer = vtk.vtkRenderer()

for vtkPointCloud in vtkPointClouds:
    renderer.AddActor(vtkPointCloud.vtkActor)

renderer.ResetCamera()

# Axes
axes = vtk.vtkAxesActor()
axes.SetTotalLength(axis_length, axis_length, axis_length)
axes.SetShaftTypeToLine()
axes.SetCylinderRadius(0.05)
axes.SetConeRadius(0.05)
axes.SetSphereRadius(0.1)

# Set axes color to white
axes.GetXAxisShaftProperty().SetColor(1, 1, 1)
axes.GetYAxisShaftProperty().SetColor(1, 1, 1)
axes.GetZAxisShaftProperty().SetColor(1, 1, 1)
axes.GetXAxisTipProperty().SetColor(1, 1, 1)
axes.GetYAxisTipProperty().SetColor(1, 1, 1)
axes.GetZAxisTipProperty().SetColor(1, 1, 1)
renderer.AddActor(axes)

# Axes labels
axes_labels = vtk.vtkCubeAxesActor()
axes_labels.SetBounds(min_vals[0], max_vals[0], min_vals[1], max_vals[1], min_vals[2], max_vals[2])
axes_labels.SetCamera(renderer.GetActiveCamera())
axes_labels.GetTitleTextProperty(0).SetColor(1, 1, 1)
axes_labels.GetTitleTextProperty(1).SetColor(1, 1, 1)
axes_labels.GetTitleTextProperty(2).SetColor(1, 1, 1)
axes_labels.GetLabelTextProperty(0).SetColor(1, 1, 1)
axes_labels.GetLabelTextProperty(1).SetColor(1, 1, 1)
axes_labels.GetLabelTextProperty(2).SetColor(1, 1, 1)
renderer.AddActor(axes_labels)

# Render Window
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(1600, 1200)

# Interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Begin Interaction
print('Beginning 3D render')
renderWindow.Render()
renderWindowInteractor.Start()
