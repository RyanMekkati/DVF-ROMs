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
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <root_folder>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string rootFolder = argv[1];
    const std::vector<std::string> phases = {"00","10","20","30","40","50","60","70","80","90"};

    using PixelType       = float;
    using VolumeType      = itk::Image<PixelType,3>;
    using VectorPixelType = itk::Vector<double,3>;
    using DVFType         = itk::Image<VectorPixelType,3>;
    using ReaderType      = itk::ImageSeriesReader<VolumeType>;
    using NamesGenType    = itk::GDCMSeriesFileNames;
    using DemonsFilterType= itk::DemonsRegistrationFilter<VolumeType,VolumeType,DVFType>;
    using ITKtoVTKFilter  = itk::ImageToVTKImageFilter<DVFType>;

    std::vector<VolumeType::Pointer> volumes;
    for (const auto& ph : phases)
    {
        std::string phasePath = rootFolder + "/" + ph;
        std::cout << "Loading phase " << ph << " from " << phasePath << std::endl;

        NamesGenType::Pointer nameGen = NamesGenType::New();
        nameGen->SetDirectory(phasePath);
        const auto& seriesUIDs = nameGen->GetSeriesUIDs();
        if (seriesUIDs.empty())
        {
            std::cerr << "  No DICOM series found in " << phasePath << std::endl;
            continue;
        }
        auto fileNames = nameGen->GetFileNames(seriesUIDs.front());

        ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileNames(fileNames);
        try { reader->Update(); }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "  Error reading " << phasePath << ": " << err << std::endl;
            continue;
        }
        volumes.push_back(reader->GetOutput());
        auto size = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
        std::cout << "  Loaded size: " << size[0] << "," << size[1] << "," << size[2] << std::endl;
    }

    if (volumes.size() < 2)
    {
        std::cerr << "Need at least two phases to compute DVF." << std::endl;
        return EXIT_FAILURE;
    }

    for (size_t t = 0; t+1 < volumes.size(); ++t)
    {
        std::cout << "Computing DVF: phase " << phases[t] << " -> " << phases[t+1] << std::endl;

        // Demons registration
        DemonsFilterType::Pointer demons = DemonsFilterType::New();
        demons->SetFixedImage(volumes[t]);
        demons->SetMovingImage(volumes[t+1]);
        demons->SetNumberOfIterations(20);
        demons->Update();
        DVFType::Pointer dvf = demons->GetOutput();

        // Convert to VTK image
        ITKtoVTKFilter::Pointer itk2vtk = ITKtoVTKFilter::New();
        itk2vtk->SetInput(dvf);
        itk2vtk->Update();
        vtkImageData* vectorField = itk2vtk->GetOutput();

        // Arrow source
        auto arrow = vtkSmartPointer<vtkArrowSource>::New();

        // Subsample points for glyphs
        auto mask = vtkSmartPointer<vtkMaskPoints>::New();
        mask->SetInputData(vectorField);
        mask->SetMaximumNumberOfPoints(20000);
        mask->RandomModeOn();
        mask->Update();

        // Glyph filter
        auto glyph = vtkSmartPointer<vtkGlyph3D>::New();
        glyph->SetSourceConnection(arrow->GetOutputPort());
        glyph->SetInputConnection(mask->GetOutputPort());
        glyph->SetVectorModeToUseVector();
        glyph->SetScaleFactor(1.0);
        glyph->Update();

        // Mapper & actor
        auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(glyph->GetOutputPort());
        auto actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);

        // Renderer & window
        auto renderer = vtkSmartPointer<vtkRenderer>::New();
        renderer->AddActor(actor);
        renderer->SetBackground(0.1, 0.1, 0.2);

        auto renWin = vtkSmartPointer<vtkRenderWindow>::New();
        renWin->AddRenderer(renderer);
        renWin->SetSize(800,600);

        auto iren = vtkSmartPointer<vtkRenderWindowInteractor>::New();
        iren->SetRenderWindow(renWin);

        std::cout << "Rendering phase " << phases[t] << " -> " << phases[t+1] << std::endl;
        renWin->Render();
        iren->Start();
    }

    return EXIT_SUCCESS;
}
