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
    int q;
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

struct vtx
{
    double x;
    double y;
    double z;
    double px;
    double py;
    double pz;
};

struct dim
{
    double mass;
    double energy;
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
    double weight;
    int rec_ndim;
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
    TFile* file = new TFile("evt_tree.root", "RECREATE");
    
    TTree* dim_tree = new TTree("dim_tree", "dimu data");
    TTree* trk_tree = new TTree("trk_tree", "dimu data");
    
    evt* event = new evt();
    dim* true_dim = new dim();
    trk* rec_trk = new trk();
    vtx* true_vtx = new vtx();
    vtx* rec_vtx = new vtx();
    
    dim_tree->Branch("event", &event);
    dim_tree->Branch("true_dim", &true_dim);
    
    trk_tree->Branch("event", &event);
    trk_tree->Branch("rec_trk", &rec_trk);
    trk_tree->Branch("true_vtx", &true_vtx);
    trk_tree->Branch("rec_vtx", &rec_vtx);
    
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
            true_dim->mass = dim_true->at(j).mom.M();
            true_dim->energy = dim_true->at(j).mom.E();
            true_dim->pt = dim_true->at(j).mom.Perp();
            true_dim->x1 = dim_true->at(j).x1;
            true_dim->x2 = dim_true->at(j).x2;
            true_dim->xf = dim_true->at(j).xF;
            true_dim->costh = dim_true->at(j).costh;
            true_dim->phi = dim_true->at(j).phi;
            dim_tree->Fill();
        }
        
        
        int ntrks = trk_true->size();
        for(int j = 0; j < ntrks; j++)
        {
            event->id = i;
            event->rec_stat = ed->rec_stat;
            event->rec_ndim = ed->n_dim_reco;
            event->weight = ed->weight;
            rec_trk->q = trk_reco->at(j).charge;
            rec_trk->x1 = trk_reco->at(j).pos_st1.X();
            rec_trk->y1 = trk_reco->at(j).pos_st1.Y();
            rec_trk->z1 = trk_reco->at(j).pos_st1.Z();
            rec_trk->px1 = trk_reco->at(j).pos_st1.Px();
            rec_trk->py1 = trk_reco->at(j).pos_st1.Py();
            rec_trk->pz1 = trk_reco->at(j).pos_st1.Pz();
            rec_trk->x3 = trk_reco->at(j).pos_st3.X();
            rec_trk->y3 = trk_reco->at(j).pos_st3.Y();
            rec_trk->z3 = trk_reco->at(j).pos_st3.Z();
            rec_trk->px3 = trk_reco->at(j).pos_st3.Px();
            rec_trk->py3 = trk_reco->at(j).pos_st3.Py();
            rec_trk->pz3 = trk_reco->at(j).pos_st3.Pz();
            
            
            true_vtx->x = trk_true->at(j).pos_vtx.X();
            true_vtx->y = trk_true->at(j).pos_vtx.Y();
            true_vtx->z = trk_true->at(j).pos_vtx.Z();
            true_vtx->px = trk_true->at(j).mom_vtx.Px();
            true_vtx->py = trk_true->at(j).mom_vtx.Py();
            true_vtx->pz = trk_true->at(j).mom_vtx.Pz();
            
            
            rec_vtx->x = trk_reco->at(j).pos_vtx.X();
            rec_vtx->y = trk_reco->at(j).pos_vtx.Y();
            rec_vtx->z = trk_reco->at(j).pos_vtx.Z();
            rec_vtx->px = trk_reco->at(j).mom_vtx.Px();
            rec_vtx->py = trk_reco->at(j).mom_vtx.Py();
            rec_vtx->pz = trk_reco->at(j).mom_vtx.Pz();
            
            trk_tree->Fill();
        }
    }
    
    dim_tree->Write();
    trk_tree->Write();
    file->Close();
    
    cout << "*** evt_tree ***" << endl;
}

int event_tree()
{
    data* D = new data();
    
    D->output();
    return 0;
}

