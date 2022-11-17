/*
events tree
*/
R__LOAD_LIBRARY(libana_sim_dst)

#include <TFile.h>
#include <TTree.h>
#include <TString.h>
#include <TLorentzVector.h>
#include <iostream>

using namespace std;

struct trk
{
    int charge;
    double x1;
    double y1;
    double z1;
    double px1;
    double py1;
    double pz1;
    double x3;
    double y3;
    double z3;
    double px3;
    double py3;
    double pz3;
};

// struct vtx
// {
//     double x;
//     double y;
//     double z;
//     double px;
//     double py;
//     double pz;
// };

struct dim
{
    double x;
    double y;
    double z;
    double px;
    double py;
    double pz;
    double mass;
    double pt;
    double x1;
    double x2;
    double xf;
    double costh;
    double phi;
};

struct evt
{
    int id;
    int rec_stat;
    int rec_ndim;
    double weight;
    dim true_dim;
    trk rec_pos_trk;
    trk rec_neg_trk;
    // vtx true_vtx;
    // vtx rec_vtx;
};

class data
{
  public:
    TFile* F1;
    TTree* T1;
    int nevents;
    EventData* ed = new EventData();
    TrackList* trk_true = new TrackList();
    TrackList* trk_reco = new TrackList();
    DimuonList* dim_true = new DimuonList();
    

    data();
    void output();
};


data::data()
{
    TFile* F1 = TFile::Open("sim_tree.root", "READ");
    T1 = (TTree*)F1->Get("tree");
    nevents = T1->GetEntries();
    
    T1->SetBranchAddress("evt", &ed);
    T1->SetBranchAddress("trk_true", &trk_true);
    T1->SetBranchAddress("trk_reco", &trk_reco);
    T1->SetBranchAddress("dim_true", &dim_true);

    cout << "*** sim_tree ***" << endl;
}

void data::output()
{
    TFile* file = new TFile("vertexml.root", "RECREATE");
    
    TTree* tree = new TTree("tree", "dimu tree");
    
    evt* event = new evt();
    
    tree->Branch("event", &event);
    
    for(int i = 0; i < nevents; i++)
    {
        T1->GetEntry(i);
        
        int ndim = dim_true->size();
        
        for(int j = 0; j < ndim; j++)
        {
            event->id = i;
            
            event->rec_ndim = ed->n_dim_reco;
            event->rec_stat = ed->rec_stat;
            event->weight = ed->weight;
            
            event->true_dim.x = dim_true->at(j).pos.X();
            event->true_dim.y = dim_true->at(j).pos.Y();
            event->true_dim.z = dim_true->at(j).pos.Z();
            event->true_dim.px = dim_true->at(j).mom.Px();
            event->true_dim.py = dim_true->at(j).mom.Py();
            event->true_dim.pz = dim_true->at(j).mom.Pz();
            
            event->true_dim.mass = dim_true->at(j).mom.M();
            event->true_dim.pt = dim_true->at(j).pT;
            event->true_dim.x1 = dim_true->at(j).x1;
            event->true_dim.x2 = dim_true->at(j).x2;
            event->true_dim.xf = dim_true->at(j).xF;
            event->true_dim.costh = dim_true->at(j).costh;
            event->true_dim.phi = dim_true->at(j).phi;
        }
        
        
        int ntrks = trk_true->size();
        for(int j = 0; j < ntrks; j++)
        {
            if(trk_reco->at(j).charge==+1)
            {
                event->rec_pos_trk.charge = trk_reco->at(j).charge;
                
                event->rec_pos_trk.x1 = trk_reco->at(j).pos_st1.X();
                event->rec_pos_trk.y1 = trk_reco->at(j).pos_st1.Y();
                event->rec_pos_trk.z1 = trk_reco->at(j).pos_st1.Z();
                event->rec_pos_trk.px1 = trk_reco->at(j).pos_st1.Px();
                event->rec_pos_trk.py1 = trk_reco->at(j).pos_st1.Py();
                event->rec_pos_trk.pz1 = trk_reco->at(j).pos_st1.Pz();
                
                event->rec_pos_trk.x3 = trk_reco->at(j).pos_st3.X();
                event->rec_pos_trk.y3 = trk_reco->at(j).pos_st3.Y();
                event->rec_pos_trk.z3 = trk_reco->at(j).pos_st3.Z();
                event->rec_pos_trk.px3 = trk_reco->at(j).pos_st3.Px();
                event->rec_pos_trk.py3 = trk_reco->at(j).pos_st3.Py();
                event->rec_pos_trk.pz3 = trk_reco->at(j).pos_st3.Pz();
            }
            
            if(trk_reco->at(j).charge==-1)
            {
                event->rec_neg_trk.charge = trk_reco->at(j).charge;
                
                event->rec_neg_trk.x1 = trk_reco->at(j).pos_st1.X();
                event->rec_neg_trk.y1 = trk_reco->at(j).pos_st1.Y();
                event->rec_neg_trk.z1 = trk_reco->at(j).pos_st1.Z();
                event->rec_neg_trk.px1 = trk_reco->at(j).pos_st1.Px();
                event->rec_neg_trk.py1 = trk_reco->at(j).pos_st1.Py();
                event->rec_neg_trk.pz1 = trk_reco->at(j).pos_st1.Pz();
                
                event->rec_neg_trk.x3 = trk_reco->at(j).pos_st3.X();
                event->rec_neg_trk.y3 = trk_reco->at(j).pos_st3.Y();
                event->rec_neg_trk.z3 = trk_reco->at(j).pos_st3.Z();
                event->rec_neg_trk.px3 = trk_reco->at(j).pos_st3.Px();
                event->rec_neg_trk.py3 = trk_reco->at(j).pos_st3.Py();
                event->rec_neg_trk.pz3 = trk_reco->at(j).pos_st3.Pz();
            }
        }
        
        tree->Fill();
    }
    
    tree->Write();
    file->Close();
    
    cout << "*** vertexml ***" << endl;
}

int events()
{
    data* D = new data();
    
    D->output();
    return 0;
}

