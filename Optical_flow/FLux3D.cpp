#include <new>  // placement new support
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include <itkImage.h>
#include <itkVector.h>
#include <itkImageSeriesReader.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkDemonsRegistrationFilter.h>
#include <itkImageToVTKImageFilter.h>
#include <itkShrinkImageFilter.h>
#include <itkRegionOfInterestImageFilter.h>

#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkArrowSource.h>
#include <vtkMaskPoints.h>
#include <vtkGlyph3D.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <root_folder> [xmin ymin zmin xsize ysize zsize]" << std::endl;
        return EXIT_FAILURE;
    }
    const std::string rootFolder = argv[1];
    bool useROI = (argc >= 8);
    itk::Index<3> roiIndex{};
    itk::Size<3> roiSize{};
    if (useROI) {
        roiIndex[0] = std::stoi(argv[2]);
        roiIndex[1] = std::stoi(argv[3]);
        roiIndex[2] = std::stoi(argv[4]);
        roiSize[0]  = std::stoi(argv[5]);
        roiSize[1]  = std::stoi(argv[6]);
        roiSize[2]  = std::stoi(argv[7]);
        std::cout << "Using ROI: index="
                  << roiIndex << " size=" << roiSize << std::endl;
    }

    const std::vector<std::string> phases =
      {"00","10","20","30","40","50","60","70","80","90"};

    using PixelType       = float;
    using VolumeType      = itk::Image<PixelType,3>;
    using VectorPixelType = itk::Vector<double,3>;
    using DVFType         = itk::Image<VectorPixelType,3>;
    using ReaderType      = itk::ImageSeriesReader<VolumeType>;
    using NamesGenType    = itk::GDCMSeriesFileNames;
    using DemonsFilterType= itk::DemonsRegistrationFilter<VolumeType,VolumeType,DVFType>;
    using ITKtoVTKFilter  = itk::ImageToVTKImageFilter<DVFType>;
    using ShrinkerType    = itk::ShrinkImageFilter<VolumeType,VolumeType>;
    using ROIFilterType   = itk::RegionOfInterestImageFilter<VolumeType,VolumeType>;

    std::vector<VolumeType::Pointer> volumes;
    for (const auto& ph : phases) {
        std::string path = rootFolder + "/" + ph;
        std::cout << "Loading phase " << ph << " from " << path << std::endl;
        NamesGenType::Pointer namer = NamesGenType::New();
        namer->SetDirectory(path);
        auto uids = namer->GetSeriesUIDs();
        if (uids.empty()) continue;
        auto files = namer->GetFileNames(uids.front());
        ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileNames(files);
        try { reader->Update(); }
        catch (itk::ExceptionObject & err) {
            std::cerr << "Error reading " << path << ": " << err << std::endl;
            continue;
        }
        VolumeType::Pointer vol = reader->GetOutput();
        if (useROI) {
            ROIFilterType::Pointer roiFilter = ROIFilterType::New();
            itk::ImageRegion<3> region;
            region.SetIndex(roiIndex);
            region.SetSize(roiSize);
            roiFilter->SetRegionOfInterest(region);
            roiFilter->SetInput(vol);
            roiFilter->Update();
            vol = roiFilter->GetOutput();
        }
        volumes.push_back(vol);
    }
    if (volumes.size() < 2) {
        std::cerr << "Need at least two phases to compute DVF." << std::endl;
        return EXIT_FAILURE;
    }

    for (size_t t = 0; t+1 < volumes.size(); ++t) {
        std::cout << "Computing DVF: " << phases[t] << "->" << phases[t+1] << std::endl;
        // Downsample
        auto shrinkF = ShrinkerType::New();
        shrinkF->SetInput(volumes[t]);
        shrinkF->SetShrinkFactors(2);  // shrink by factor 2 in all dimensions
        shrinkF->Update();
        auto shrinkM = ShrinkerType::New();
        shrinkM->SetInput(volumes[t+1]);
        shrinkM->SetShrinkFactors(2);  // shrink by factor 2 in all dimensions
        shrinkM->Update();
        // Coarse Demons
        DemonsFilterType::Pointer demons = DemonsFilterType::New();
        demons->SetFixedImage(shrinkF->GetOutput());
        demons->SetMovingImage(shrinkM->GetOutput());
        demons->SetNumberOfIterations(50);
        demons->Update();
        // Full Demons
        demons = DemonsFilterType::New();
        demons->SetFixedImage(volumes[t]);
        demons->SetMovingImage(volumes[t+1]);
        demons->SetNumberOfIterations(20);
        demons->Update();
        DVFType::Pointer dvf = demons->GetOutput();
        // To VTK
        ITKtoVTKFilter::Pointer itk2vtk = ITKtoVTKFilter::New();
        itk2vtk->SetInput(dvf); itk2vtk->Update();
        vtkImageData* vec = itk2vtk->GetOutput();
        // Glyphs
        auto arrow = vtkSmartPointer<vtkArrowSource>::New();
        auto mask = vtkSmartPointer<vtkMaskPoints>::New();
        mask->SetInputData(vec); mask->SetMaximumNumberOfPoints(10000); mask->RandomModeOn(); mask->Update();
        auto glyph = vtkSmartPointer<vtkGlyph3D>::New();
        glyph->SetSourceConnection(arrow->GetOutputPort()); glyph->SetInputConnection(mask->GetOutputPort());
        glyph->SetVectorModeToUseVector(); glyph->SetScaleFactor(1.0); glyph->Update();
        auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(glyph->GetOutputPort());
        auto actor = vtkSmartPointer<vtkActor>::New(); actor->SetMapper(mapper);
        auto renderer = vtkSmartPointer<vtkRenderer>::New();
        renderer->AddActor(actor); renderer->SetBackground(0.1,0.1,0.2);
        auto renWin = vtkSmartPointer<vtkRenderWindow>::New(); renWin->AddRenderer(renderer);
        auto iren = vtkSmartPointer<vtkRenderWindowInteractor>::New(); iren->SetRenderWindow(renWin);
        renWin->Render(); iren->Start();
    }
    return EXIT_SUCCESS;
}

// Compile line: cd ~/Documents/GitHub/DVF-Algorithms-2D-3D/Optical_flow
// rm -rf build && mkdir build && cd build
// cmake ..
// make
// ./optical_flow_3d /home/ryan/Desktop/4DCT-Dicom_P1 \
    100 120 40  64 64 32