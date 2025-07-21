#include <new>  // placement new support
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <sys/stat.h>
#include <filesystem>
#include <regex>
#include <map>
#include <algorithm>
#include <fstream>

#include <itkImage.h>
#include <itkVector.h>
#include <itkImageSeriesReader.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageFileReader.h>
#include <itkDemonsRegistrationFilter.h>
#include <itkImageToVTKImageFilter.h>
#include <itkShrinkImageFilter.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionConstIterator.h>

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

// Utility to check if a path is a directory
bool IsDirectory(const std::string &path) {
    struct stat s;
    if (stat(path.c_str(), &s) == 0) {
        return (s.st_mode & S_IFDIR) != 0;
    }
    return false;
}

// Output root for DVFs
const char* kDVFOutRoot = "/home/ryan/Documents/GitHub/DVF-Algorithms-2D-3D/DVF-data";

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <root_folder_or_image> [xmin ymin zmin xsize ysize zsize]" << std::endl;
        return EXIT_FAILURE;
    }
    const std::string rootFolder = argv[1];

    // Parse optional ROI arguments
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
        std::cout << "Using ROI index=" << roiIndex << " size=" << roiSize << std::endl;
    }

    // ITK typedefs
    using PixelType        = float;
    using VolumeType       = itk::Image<PixelType,3>;
    using VectorPixelType  = itk::Vector<double,3>;
    using DVFType          = itk::Image<VectorPixelType,3>;
    using SeriesReaderType = itk::ImageSeriesReader<VolumeType>;
    using NamesGenType     = itk::GDCMSeriesFileNames;
    using FileReaderType   = itk::ImageFileReader<VolumeType>;
    using DemonsFilterType = itk::DemonsRegistrationFilter<VolumeType,VolumeType,DVFType>;
    using ITKtoVTKFilter   = itk::ImageToVTKImageFilter<DVFType>;
    using ShrinkerType     = itk::ShrinkImageFilter<VolumeType,VolumeType>;
    using ROIFilterType    = itk::RegionOfInterestImageFilter<VolumeType,VolumeType>;

    // Standard phase labels for fallback
    const std::vector<std::string> phases =
      {"00","10","20","30","40","50","60","70","80","90"};

    namespace fs = std::filesystem;

    // Setup output directories and manifest
    fs::path rootPath(rootFolder);
    std::string patientID = rootPath.filename().string();
    if (patientID.empty()) patientID = rootPath.parent_path().filename().string();
    fs::path outRoot(kDVFOutRoot);
    fs::create_directories(outRoot / patientID);
    fs::path manifestPath = outRoot / "dvf_manifest.csv";
    std::ofstream manifestOfs(manifestPath, std::ios::app);
    if (manifestOfs.tellp() == 0) manifestOfs << "patient,phase_from,phase_to,method,path_nifti\n";
    const std::string methodTag = "DEMONS";

    //--------------------------------------------------------------------
    // Discover input series: group .mha by prefix_phN_masked or fallback to DICOM phase-folders
    std::vector<std::vector<std::string>> allSeriesInputs;
    const std::regex mhaRe(R"(^(.+?)_ph(\d+)_masked\.(mha|mhd)$)", std::regex::icase);

    if (!fs::is_directory(rootFolder)) {
        // Single file input
        allSeriesInputs.push_back({ rootFolder });
    }
    else {
        // Try grouping MHA series
        std::map<std::string, std::vector<std::pair<int,std::string>>> groups;
        std::smatch m;
        for (auto &entry : fs::directory_iterator(rootFolder)) {
            if (!entry.is_regular_file()) continue;
            auto fname = entry.path().filename().string();
            if (std::regex_match(fname, m, mhaRe)) {
                groups[m[1].str()].emplace_back(
                  std::stoi(m[2].str()), entry.path().string());
            }
        }
        if (!groups.empty()) {
            // Sort and collect each MHA series
            for (auto &kv : groups) {
                auto &vec = kv.second;
                std::sort(vec.begin(), vec.end(),[](auto &a, auto &b){return a.first<b.first;});
                std::vector<std::string> series;
                for (auto &p : vec) series.push_back(p.second);
                allSeriesInputs.push_back(std::move(series));
            }
        }
        else {
            // Fallback: DICOM phase subfolders
            std::vector<std::string> dirs;
            for (auto &ph : phases) {
                std::string d = rootFolder + "/" + ph;
                if (fs::is_directory(d)) dirs.push_back(d);
            }
            allSeriesInputs.push_back(std::move(dirs));
        }
    }
    //--------------------------------------------------------------------

    // Process each discovered series
    for (auto &seriesFiles : allSeriesInputs) {
        // Load volumes
        std::vector<VolumeType::Pointer> volumes;
        for (auto &inp : seriesFiles) {
            std::cout << "Loading " << inp << std::endl;
            VolumeType::Pointer vol;
            if (IsDirectory(inp)) {
                NamesGenType::Pointer namer = NamesGenType::New();
                namer->SetUseSeriesDetails(true);
                namer->SetDirectory(inp);
                auto uids = namer->GetSeriesUIDs();
                if (uids.empty()) continue;
                auto files = namer->GetFileNames(uids.front());
                SeriesReaderType::Pointer sr = SeriesReaderType::New();
                sr->SetFileNames(files); sr->Update();
                vol = sr->GetOutput();
            } else {
                FileReaderType::Pointer fr = FileReaderType::New();
                fr->SetFileName(inp); fr->Update(); vol = fr->GetOutput();
            }
            if (useROI) {
                ROIFilterType::Pointer roi = ROIFilterType::New();
                roi->SetInput(vol);
                roi->SetRegionOfInterest(itk::ImageRegion<3>(roiIndex,roiSize));
                roi->Update(); vol = roi->GetOutput();
            }
            volumes.push_back(vol);
        }

        // Compute DVFs
        for (size_t t = 0; t + 1 < volumes.size(); ++t) {
            std::cout << "DVF " << t << "->" << (t+1) << std::endl;
            auto sF = ShrinkerType::New(); sF->SetInput(volumes[t]); sF->SetShrinkFactors(2); sF->Update();
            auto sM = ShrinkerType::New(); sM->SetInput(volumes[t+1]); sM->SetShrinkFactors(2); sM->Update();
            DemonsFilterType::Pointer d = DemonsFilterType::New();
            d->SetFixedImage(sF->GetOutput()); d->SetMovingImage(sM->GetOutput()); d->SetNumberOfIterations(50); d->Update();
            d = DemonsFilterType::New(); d->SetFixedImage(volumes[t]); d->SetMovingImage(volumes[t+1]); d->SetNumberOfIterations(20); d->Update();
            DVFType::Pointer dvf = d->GetOutput();

            // Save DVF
            using Writer = itk::ImageFileWriter<DVFType>;
            Writer::Pointer writer = Writer::New();
            std::ostringstream oss;
            oss << patientID << "_P" << t << "toP" << (t+1) << "_dvf_" << methodTag << ".nii.gz";
            auto out = outRoot / patientID / oss.str();
            writer->SetFileName(out.string()); writer->SetInput(dvf); writer->UseCompressionOn(); writer->Update();
            manifestOfs << patientID << "," << t << "," << (t+1) << "," << methodTag << "," << out.string() << "\n";

            // VTK visualization
            ITKtoVTKFilter::Pointer itv = ITKtoVTKFilter::New(); itv->SetInput(dvf); itv->Update();
            vtkImageData* vec = itv->GetOutput();
            auto arrow = vtkSmartPointer<vtkArrowSource>::New();
            auto mask = vtkSmartPointer<vtkMaskPoints>::New(); mask->SetInputData(vec); mask->SetMaximumNumberOfPoints(10000); mask->RandomModeOn(); mask->Update();
            auto glyph = vtkSmartPointer<vtkGlyph3D>::New(); glyph->SetSourceConnection(arrow->GetOutputPort()); glyph->SetInputConnection(mask->GetOutputPort()); glyph->SetVectorModeToUseVector(); glyph->SetScaleFactor(1.0); glyph->Update();
            auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New(); mapper->SetInputConnection(glyph->GetOutputPort());
            auto actor = vtkSmartPointer<vtkActor>::New(); actor->SetMapper(mapper);
            auto ren = vtkSmartPointer<vtkRenderer>::New(); ren->AddActor(actor); ren->SetBackground(0.1,0.1,0.2);
            auto win = vtkSmartPointer<vtkRenderWindow>::New(); win->AddRenderer(ren);
            auto ir = vtkSmartPointer<vtkRenderWindowInteractor>::New(); ir->SetRenderWindow(win); win->Render(); ir->Start();
        }
    }

    return EXIT_SUCCESS;
}

// Compile line: cd ~/Documents/GitHub/DVF-Algorithms-2D-3D/Optical_flow
// rm -rf build && mkdir build && cd build
// cmake ..
// make
// ./optical_flow_3d /home/ryan/Desktop/4DCT-Dicom_P1 \
    100 120 40  64 64 32
