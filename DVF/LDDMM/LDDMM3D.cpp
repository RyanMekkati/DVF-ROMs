#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include <itkImage.h>
#include <itkVector.h>
#include <itkImageSeriesReader.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkDiffeomorphicDemonsRegistrationFilter.h>
#include <itkRegionOfInterestImageFilter.h>

int main(int argc, char* argv[])
{
  if(argc < 8) {
    std::cerr << "Usage: " << argv[0]
              << " <DICOMRoot> x0 y0 z0 sx sy sz [iterations]" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string root = argv[1];
  itk::Index<3> roiIndex;
  itk::Size<3>  roiSize;
  for(int i=0; i<3; ++i) roiIndex[i] = std::stoi(argv[2+i]);
  for(int i=0; i<3; ++i) roiSize[i]  = std::stoi(argv[5+i]);
  unsigned int iterations = (argc>8) ? std::stoi(argv[8]) : 50;

  using PixelType = float;
  constexpr unsigned int Dim = 3;
  using ImageType = itk::Image<PixelType,Dim>;
  using VectorType = itk::Vector<PixelType,Dim>;
  using FieldType = itk::Image<VectorType,Dim>;

  itk::GDCMSeriesFileNames::Pointer names = itk::GDCMSeriesFileNames::New();
  names->SetDirectory(root);
  std::vector<std::string> phases = {"00","10","20","30","40","50","60","70","80","90"};

  for(size_t p=1; p<phases.size(); ++p) {
    auto fixedDir  = root + "/" + phases[p-1];
    auto movingDir = root + "/" + phases[p];
    names->SetDirectory(fixedDir);
    auto fixedFiles = names->GetInputFileNames();
    names->SetDirectory(movingDir);
    auto movingFiles = names->GetInputFileNames();

    auto fixedReader  = itk::ImageSeriesReader<ImageType>::New();
    auto movingReader = itk::ImageSeriesReader<ImageType>::New();
    fixedReader->SetFileNames(fixedFiles);
    movingReader->SetFileNames(movingFiles);

    auto roiF = itk::RegionOfInterestImageFilter<ImageType,ImageType>::New();
    roiF->SetRegionOfInterest({roiIndex,roiSize});
    roiF->SetInput(fixedReader->GetOutput());
    auto roiM = itk::RegionOfInterestImageFilter<ImageType,ImageType>::New();
    roiM->SetRegionOfInterest({roiIndex,roiSize});
    roiM->SetInput(movingReader->GetOutput());

    using DemonsFilter = itk::DiffeomorphicDemonsRegistrationFilter<ImageType,ImageType,FieldType>;
    auto demons = DemonsFilter::New();
    demons->SetFixedImage(roiF->GetOutput());
    demons->SetMovingImage(roiM->GetOutput());
    demons->SetNumberOfIterations(iterations);
    demons->SetStandardDeviations(1.0);
    try { demons->Update(); }
    catch(const itk::ExceptionObject & e) {
      std::cerr << "Error: " << e << std::endl;
      return EXIT_FAILURE;
    }

    auto dvf = demons->GetDisplacementField();
    // TODO: hand off dvf to your VTK visualization pipeline
  }

  return EXIT_SUCCESS;
}


