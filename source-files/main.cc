//
// make simple data file
//

R__LOAD_LIBRARY(libana_sim_dst)

#include <TFile.h>
#include <TTree.h>
#include <TString.h>
#include <TLorentzVector.h>
#include <iostream>
#include <fstream>

using namespace std;

class data
{
  public:
    TFile* file;
    TTree* tree;
    int nentries;
    EventData* ed = new EventData();
    TrackList* trk_true = new TrackList();
    TrackList* trk_reco = new TrackList();

    ofstream out_file;
//    ofstream target_file;

    data();
    void debug();
    void print();
};

data::data()
{
  TFile* file = TFile::Open("sim_tree.root", "READ");
  tree = (TTree*)file->Get("tree");
  nentries = tree->GetEntries();
  //
  //
  tree->SetBranchAddress("evt", &ed);
  tree->SetBranchAddress("trk_true", &trk_true);
  tree->SetBranchAddress("trk_reco", &trk_reco);
  //
  //
}

void data::debug()
{
  for(int i = 0; i < 20; i++)
  {
    tree->GetEntry(i);
    int ntracks = trk_true->size();
    for(int j = 0; j < ntracks; j++)
    {
      if(trk_true->at(j).charge == +1)
      {
        cout << "charge : " << trk_true->at(j).charge << " px_st1 : " << trk_true->at(j).mom_st1.Px() << " py_st1 : " << trk_true->at(j).mom_st1.Py() << endl;
      }
    }
  }
}

void data::print()
{
  out_file.open("raw_data.csv");
  out_file << "q1" << "," << "x1" << "," << "y1" << "," << "z1" << "," << "px1" << "," << "py1" << "," << "pz1" << ","
  << "x2" << "," << "y2" << "," << "z2" << "," << "px2" << "," << "py2" << "," << "pz2" << ","
  << "q2" << "," <<  "vtx" << "," << "vty" << "," << "vtz" << "," << "vpx" << "," << "vpy" << "," << "vpz" << endl;


  for(int i = 0; i < nentries; i++)
  {
    tree->GetEntry(i);
    if(ed->rec_stat == 0)
    {
      int ntracks = trk_true->size();
      for(int j = 0; j < ntracks; j++)
      {
        out_file << trk_reco->at(j).charge << "," << trk_reco->at(j).pos_st1.X() << "," << trk_reco->at(j).pos_st1.Y() << "," << trk_reco->at(j).pos_st1.Z()
        << "," << trk_reco->at(j).mom_st1.Px() << "," << trk_reco->at(j).mom_st1.Py() << "," << trk_reco->at(j).mom_st1.Pz() << ","
        << trk_reco->at(j).pos_st3.X() << "," << trk_reco->at(j).pos_st3.Y() << "," << trk_reco->at(j).pos_st3.Z()
        << "," << trk_reco->at(j).mom_st3.Px() << "," << trk_reco->at(j).mom_st3.Py() << "," << trk_reco->at(j).mom_st3.Pz() << ","
        << trk_true->at(j).charge << "," << trk_true->at(j).pos_vtx.X() << "," << trk_true->at(j).pos_vtx.Y() << "," << trk_true->at(j).pos_vtx.Z()
        << "," << trk_true->at(j).mom_vtx.Px() << "," << trk_true->at(j).mom_vtx.Py() << "," << trk_true->at(j).mom_vtx.Pz() << endl;
      }
    }
  }
  out_file.close();
}

int main()
{
  data* D = new data();
//  D->debug();
  D->print();
  return 0;
}