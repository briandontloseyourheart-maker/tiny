#include "myfunc.H"

namespace {
    void GotoNextLine (std::istream& is)
    {
        constexpr std::streamsize bl_ignore_max { 100000 };
        is.ignore(bl_ignore_max, '\n');
    }
}

// create a checkpoint directory
// write out time and BoxArray to a Header file
// write out multifab data
void WriteCheckpoint(const int& step,
                     const Real& time,
                     const MultiFab& U)
{
    // checkpoint file name, e.g., chk00010
    const std::string& checkpointname = Concatenate("./checkpoint/chk",step,5);

    BoxArray ba = U.boxArray();

    // single level problem
    int nlevels = 1;

    // ---- prebuild a hierarchy of directories
    // ---- dirName is built first.  if dirName exists, it is renamed.  then build
    // ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
    // ---- if callBarrier is true, call ParallelDescriptor::Barrier()
    // ---- after all directories are built
    // ---- ParallelDescriptor::IOProcessor() creates the directories
    PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);

    VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

    // write Header file to store time and BoxArray
    if (ParallelDescriptor::IOProcessor()) {

        std::ofstream HeaderFile;
        HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
        std::string HeaderFileName(checkpointname + "/Header");
        HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out   |
                        std::ofstream::trunc |
                        std::ofstream::binary);

        if( !HeaderFile.good()) {
            FileOpenFailed(HeaderFileName);
        }

        HeaderFile.precision(17);

        // write out title line
        HeaderFile << "Checkpoint file for MagneX\n";

        // write out time
        HeaderFile << time << "\n";

        // write the BoxArray
        ba.writeOn(HeaderFile);
        HeaderFile << '\n';
    }

    // write the MultiFab data to, e.g., chk00010/Level_0/
    VisMF::Write(U, MultiFabFileFullPrefix(0, checkpointname, "Level_", "U"));
}

// read in the time and BoxArray, then create a DistributionMapping
// Define phi and fill it with data from the checkpoint file
void ReadCheckpoint(const int& restart,
                    Real& time,
                    MultiFab& U,
                    BoxArray& ba,
                    DistributionMapping& dm,
                    int Nghost)
{
    // checkpoint file name, e.g., chk00010
    const std::string& checkpointname = Concatenate("./checkpoint/chk",restart,5);

    Print() << "Restart from checkpoint " << checkpointname << "\n";

    VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

    std::string line, word;

    // Header
    {
        std::string File(checkpointname + "/Header");
        Vector<char> fileCharPtr;
        ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
        std::string fileCharPtrString(fileCharPtr.dataPtr());
        std::istringstream is(fileCharPtrString, std::istringstream::in);

        // read in title line
        std::getline(is, line);

        // read in time
        is >> time;
        GotoNextLine(is);

        // read in BoxArray from Header
        ba.readFrom(is);
        GotoNextLine(is);

        // create a distribution mapping
        dm.define(ba, ParallelDescriptor::NProcs());

        U.define(ba, dm, NSTATE, Nghost);
    }

    // read in the MultiFab data
    VisMF::Read(U, MultiFabFileFullPrefix(0, checkpointname, "Level_", "U"));

}
