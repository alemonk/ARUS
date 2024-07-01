import numpy as np
import vtk
import random

# Filenames for the point clouds of different classes
pointcloud_filename_class0 = 'recon/pointclouds/class_0.txt'
pointcloud_filename_class1 = 'recon/pointclouds/class_1.txt'

class VtkPointCloud:
    def __init__(self, zMin=0, zMax=0, maxNumPoints=1e6, pointSize=1):
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
lines_class0 = np.loadtxt(pointcloud_filename_class0, delimiter=',', unpack=False)
lines_class1 = np.loadtxt(pointcloud_filename_class1, delimiter=',', unpack=False)

# Compute the bounding box of the point clouds
all_lines = np.concatenate((lines_class0, lines_class1), axis=0)
min_vals = np.min(all_lines, axis=0)
max_vals = np.max(all_lines, axis=0)
axis_length = max((max_vals[0] - min_vals[0]) * 1.5, (max_vals[1] - min_vals[1]) * 1.5, (max_vals[2] - min_vals[2]) * 1.5)

# Create VTK point clouds with different colors
vtkPointCloud_class0 = VtkPointCloud(zMin=min_vals[2], zMax=max_vals[2])
vtkPointCloud_class1 = VtkPointCloud(zMin=min_vals[2], zMax=max_vals[2])

# Add points to the point clouds
for point in lines_class0:
    vtkPointCloud_class0.addPoint(point, (200, 200, 200))

for point in lines_class1:
    vtkPointCloud_class1.addPoint(point, (255, 192, 203))

# Renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(vtkPointCloud_class0.vtkActor)
renderer.AddActor(vtkPointCloud_class1.vtkActor)
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
