#include <itkImage.h>
#include <itkVector.h>
#include <itkImageSeriesReader.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageFileReader.h>
// ITK 5.3 uses "StationaryVelocityFieldTransformExpander" header
#include <itkExponentialDisplacementFieldImageFilter.h> // scaling-and-squaring expander
#include <itkDisplacementFieldTransform.h>
#include <itkResampleImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkImageToVTKImageFilter.h>

#include <vtkSmartPointer.h>
#include <vtkMaskPoints.h>
#include <vtkArrowSource.h>
#include <vtkGlyph3D.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include <iostream>
#include <string>

using PixelType = float;
using Image3D    = itk::Image<PixelType,3>;
using VectorType = itk::Vector<PixelType,3>;
using SVF3D      = itk::Image<VectorType,3>;

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <fixedDICOMDir> <movingDICOMDir> <svfFile>\n";
    return EXIT_FAILURE;
  }

  const std::string fixedDir  = argv[1];
  const std::string movingDir = argv[2];
  const std::string svfFile   = argv[3];

  // 1) Read fixed volume
  using ReaderType = itk::ImageSeriesReader<Image3D>;
  using NamesGeneratorType = itk::GDCMSeriesFileNames;
  auto nameGenFixed = NamesGeneratorType::New();
  nameGenFixed->SetDirectory(fixedDir);
  const auto &seriesUIDsFixed = nameGenFixed->GetSeriesUIDs();
  auto namesFixed = nameGenFixed->GetFileNames(seriesUIDsFixed[0]);
  auto readerFixed = ReaderType::New();
  readerFixed->SetFileNames(namesFixed);
  readerFixed->Update();
  auto fixed = readerFixed->GetOutput();

  // 2) Read moving volume
  auto nameGenMoving = NamesGeneratorType::New();
  nameGenMoving->SetDirectory(movingDir);
  const auto &seriesUIDsMoving = nameGenMoving->GetSeriesUIDs();
  auto namesMoving = nameGenMoving->GetFileNames(seriesUIDsMoving[0]);
  auto readerMoving = ReaderType::New();
  readerMoving->SetFileNames(namesMoving);
  readerMoving->Update();
  auto moving = readerMoving->GetOutput();

  // 3) Load SVF (stationary velocity field)
  using SVFReaderType = itk::ImageFileReader<SVF3D>;
  auto svfReader = SVFReaderType::New();
  svfReader->SetFileName(svfFile);
  svfReader->Update();
  auto svf = svfReader->GetOutput();

  // 4) Exponentiate SVF -> phi via scaling-and-squaring
using ExpanderType = itk::ExponentialDisplacementFieldImageFilter<SVF3D, SVF3D>;
auto expander = ExpanderType::New();
expander->SetInput(svf);
expander->SetNumberOfIntegrationSteps(1 << 6); // 2^6 steps
expander->Update();
auto phi = expander->GetOutput();

  // 5) Build DisplacementFieldTransform
  using DoubleVectorImage = itk::Image<itk::Vector<double,3>,3>;
  using CastFilterType = itk::CastImageFilter<SVF3D, DoubleVectorImage>;
  auto castFilter = CastFilterType::New();
  castFilter->SetInput(phi);
  castFilter->Update();

  using TransformType = itk::DisplacementFieldTransform<double,3>;
  auto dispTx = TransformType::New();
  dispTx->SetDisplacementField(castFilter->GetOutput());

  // 6) Warp moving -> fixed
  using ResamplerType = itk::ResampleImageFilter<Image3D,Image3D>;
  auto resampler = ResamplerType::New();
  resampler->SetInput(moving);
  resampler->SetTransform(dispTx);
  resampler->SetSize(fixed->GetLargestPossibleRegion().GetSize());
  resampler->SetOutputOrigin(fixed->GetOrigin());
  resampler->SetOutputSpacing(fixed->GetSpacing());
  resampler->SetOutputDirection(fixed->GetDirection());
  resampler->SetDefaultPixelValue(0);
  resampler->Update();
  auto warped = resampler->GetOutput();

  // 7) Convert DVF (phi) to VTK and display sparse glyphs
  using VTKConverterType = itk::ImageToVTKImageFilter<SVF3D>;
  auto vtkConverter = VTKConverterType::New();
  vtkConverter->SetInput(phi);
  vtkConverter->Update();
  auto vtkImage = vtkConverter->GetOutput();

  auto maskPoints = vtkSmartPointer<vtkMaskPoints>::New();
  maskPoints->SetInputData(vtkImage);
  maskPoints->SetOnRatio(100);
  maskPoints->RandomModeOff();
  maskPoints->Update();

  auto arrow = vtkSmartPointer<vtkArrowSource>::New();
  auto glyph = vtkSmartPointer<vtkGlyph3D>::New();
  glyph->SetSourceConnection(arrow->GetOutputPort());
  glyph->SetInputConnection(maskPoints->GetOutputPort());
  glyph->SetVectorModeToUseVector();
  glyph->SetScaleFactor(1.0);
  glyph->Update();

  auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(glyph->GetOutputPort());
  auto actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);

  auto renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(actor);
  renderer->SetBackground(0.1, 0.2, 0.3);

  auto renWin = vtkSmartPointer<vtkRenderWindow>::New();
  renWin->AddRenderer(renderer);
  auto iren = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  iren->SetRenderWindow(renWin);
  renWin->Render();
  iren->Start();

  return EXIT_SUCCESS;
}