import numpy as np
import vtk
import random

pointcloud_filename = 'recon/pointcloud.txt'

# Numpy array to vtk points
def numpyArr2vtkPoints(npArray):
    vtkpoints = vtk.vtkPoints()    
    for i in range(npArray.shape[0]):
        vtkpoints.InsertNextPoint(npArray[i])       
    return vtkpoints

class VtkPointCloud:
    def __init__(self, zMin=0, zMax=0, maxNumPoints=1e6, pointSize=4):
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
        if zMin==0 and zMax==0:
            mapper.SetScalarVisibility(0)
            self.vtkActor.GetProperty().SetColor(0.9, 0.9, 0.9)
        else:
            mapper.SetScalarVisibility(1)
 
    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])

            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()
 
    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

lines = np.loadtxt(pointcloud_filename, delimiter=',', unpack=False)

# Compute the bounding box of the point cloud
min_vals = np.min(lines, axis=0)
max_vals = np.max(lines, axis=0)
axis_length = max( (max_vals[0] - min_vals[0]) * 1.5, (max_vals[1] - min_vals[1]) * 1.5, (max_vals[2] - min_vals[2]) * 1.5 )

vtkPointCloud = VtkPointCloud(zMin=min_vals[2], zMax=max_vals[2])
# vtkPointCloud = VtkPointCloud()
for i in lines:
    vtkPointCloud.addPoint(i)

# Renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(vtkPointCloud.vtkActor)
# renderer.SetBackground(1, 1, 1)
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
