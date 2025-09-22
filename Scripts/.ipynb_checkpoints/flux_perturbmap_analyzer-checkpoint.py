#!/usr/bin/env python3
"""
Flux PerturbMap Connection Analyzer
===================================

Simplified analyzer that identifies SHAP connections containing GenKI genes
and creates tables of unique GenKI genes and their connected molecules/metabolites.

Features:
- Finds all SHAP connections where GenKI genes appear in Source OR Target
- Case-insensitive gene name matching
- Creates unique tables for GenKI genes and connected molecules
- No ground truth validation (removed as requested)
- Timestamped outputs

Usage:
    python flux_perturbmap_analyzer.py --shap_file shap.csv --genki_file genki.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Set
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_cross_modal_dir(base_path="../Task2_CMP", pattern="cross_modal_flux_analysis_*"):
    """Find the most recent cross-modal analysis directory"""
    import glob
    import os
    
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Base cross-modal path does not exist: {base_path}")
        return None
    
    # Search for cross-modal directories with different patterns
    search_patterns = [
        str(base_path / pattern),
        str(base_path / "cross_modal_*flux*"),
        str(base_path / "cross_modal_*analysis*"),
    ]
    
    all_dirs = []
    for search_pattern in search_patterns:
        matching_dirs = glob.glob(search_pattern)
        all_dirs.extend([d for d in matching_dirs if os.path.isdir(d)])
    
    if not all_dirs:
        print(f"No cross-modal directories found in {base_path}")
        print("Searched patterns:")
        for pattern in search_patterns:
            print(f"  {pattern}")
        return None
    
    # Remove duplicates and sort by modification time
    all_dirs = list(set(all_dirs))
    all_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"Found {len(all_dirs)} cross-modal directories:")
    for i, analysis_dir in enumerate(all_dirs[:5]):  # Show top 5
        mtime = datetime.fromtimestamp(os.path.getmtime(analysis_dir))
        print(f"  {i+1}. {Path(analysis_dir).name} (modified: {mtime})")
    
    return all_dirs[0]


class FluxPerturbMapAnalyzer:
    """Simplified analyzer for GenKI-SHAP connections with flux modality"""
    
    def __init__(self, shap_file: str, genki_file: str):
        self.shap_file = Path(shap_file)
        self.genki_file = Path(genki_file)
        
        # Data containers
        self.shap_data = None
        self.genki_data = None
        
        # Analysis results
        self.filtered_connections = []
        self.unique_genki_genes = []
        self.unique_connected_molecules = []
        self.connection_summary = {}
        
        logger.info(f"Initialized FluxPerturbMapAnalyzer")
    
    def load_data(self) -> bool:
        """Load SHAP and GenKI files"""
        try:
            # Load SHAP data
            logger.info(f"Loading SHAP data from {self.shap_file}")
            self.shap_data = pd.read_csv(self.shap_file)
            required_shap_cols = ['Cluster', 'Direction', 'Target', 'Source', 'Value']
            if not all(col in self.shap_data.columns for col in required_shap_cols):
                raise ValueError(f"SHAP file missing required columns: {required_shap_cols}")
            
            # Load GenKI data
            logger.info(f"Loading GenKI data from {self.genki_file}")
            self.genki_data = pd.read_csv(self.genki_file)
            
            # Handle the gene name column
            if self.genki_data.columns[0] in ['Unnamed: 0', '']:
                self.genki_data = self.genki_data.rename(columns={self.genki_data.columns[0]: 'gene'})
            elif 'KO_gene' in self.genki_data.columns:
                # Your file uses KO_gene column
                self.genki_data = self.genki_data.rename(columns={'KO_gene': 'gene'})
            elif 'gene' not in self.genki_data.columns:
                # Assume first column contains gene names
                self.genki_data = self.genki_data.rename(columns={self.genki_data.columns[0]: 'gene'})
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"  SHAP: {len(self.shap_data)} entries")
            logger.info(f"  GenKI: {len(self.genki_data)} genes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def filter_shap_connections(self) -> List[Dict]:
        """Filter SHAP connections that contain GenKI genes in Source OR Target"""
        logger.info("Filtering SHAP connections containing GenKI genes...")
        
        # Get GenKI genes (uppercase for case-insensitive matching)
        genki_genes = self.genki_data['gene'].dropna().str.strip().tolist()
        genki_genes_upper = [gene.upper() for gene in genki_genes]
        
        logger.info(f"GenKI genes to search for: {len(genki_genes)}")
        logger.info(f"Sample GenKI genes: {genki_genes[:5]}")
        
        # Convert SHAP sources and targets to uppercase for matching
        self.shap_data['Source_upper'] = self.shap_data['Source'].str.strip().str.upper()
        self.shap_data['Target_upper'] = self.shap_data['Target'].str.strip().str.upper()
        
        # Filter SHAP data for rows containing GenKI genes
        filtered_rows = []
        
        for idx, row in self.shap_data.iterrows():
            source_upper = row['Source_upper']
            target_upper = row['Target_upper']
            
            # Check if either source or target contains a GenKI gene
            genki_in_source = source_upper in genki_genes_upper
            genki_in_target = target_upper in genki_genes_upper
            
            if genki_in_source or genki_in_target:
                # Identify which GenKI gene(s) are present
                matching_genki_genes = []
                
                if genki_in_source:
                    genki_idx = genki_genes_upper.index(source_upper)
                    matching_genki_genes.append({
                        'genki_gene': genki_genes[genki_idx],
                        'position': 'source',
                        'shap_name': row['Source']
                    })
                
                if genki_in_target:
                    genki_idx = genki_genes_upper.index(target_upper)
                    matching_genki_genes.append({
                        'genki_gene': genki_genes[genki_idx],
                        'position': 'target', 
                        'shap_name': row['Target']
                    })
                
                filtered_row = {
                    'cluster': row['Cluster'],
                    'direction': row['Direction'],
                    'source': row['Source'],
                    'target': row['Target'],
                    'shap_value': row['Value'],
                    'matching_genki_genes': matching_genki_genes,
                    'genki_positions': [mg['position'] for mg in matching_genki_genes]
                }
                
                filtered_rows.append(filtered_row)
        
        self.filtered_connections = filtered_rows
        
        logger.info(f"Found {len(filtered_rows)} SHAP connections containing GenKI genes")
        logger.info(f"  Original SHAP entries: {len(self.shap_data)}")
        logger.info(f"  Filtered entries: {len(filtered_rows)} ({len(filtered_rows)/len(self.shap_data)*100:.1f}%)")
        
        # Log breakdown by position
        source_connections = len([row for row in filtered_rows if 'source' in row['genki_positions']])
        target_connections = len([row for row in filtered_rows if 'target' in row['genki_positions']])
        both_connections = len([row for row in filtered_rows if 'source' in row['genki_positions'] and 'target' in row['genki_positions']])
        
        logger.info(f"  GenKI in source: {source_connections}")
        logger.info(f"  GenKI in target: {target_connections}")
        logger.info(f"  GenKI in both: {both_connections}")
        
        return filtered_rows
    
    def extract_unique_genki_genes(self) -> List[Dict]:
        """Extract unique GenKI genes found in SHAP connections"""
        logger.info("Extracting unique GenKI genes from filtered connections...")
        
        genki_gene_info = {}
        
        # Collect all GenKI genes from filtered connections
        for connection in self.filtered_connections:
            for matching_gene in connection['matching_genki_genes']:
                genki_gene = matching_gene['genki_gene']
                
                if genki_gene not in genki_gene_info:
                    # Get additional info from GenKI data
                    genki_row = self.genki_data[self.genki_data['gene'].str.strip().str.upper() == genki_gene.upper()]
                    if not genki_row.empty:
                        genki_data = genki_row.iloc[0].to_dict()
                    else:
                        genki_data = {'gene': genki_gene}
                    
                    genki_gene_info[genki_gene] = {
                        'gene_name': genki_gene,
                        'genki_data': genki_data,
                        'genki_rank': genki_data.get('gene', ''),  # Will be filled below
                        'found_as_source': 0,
                        'found_as_target': 0,
                        'total_connections': 0,
                        'directions': set(),
                        'clusters': set(),
                        'avg_shap_value': 0,
                        'max_shap_value': 0,
                        'min_shap_value': float('inf')
                    }
                
                # Update statistics
                gene_info = genki_gene_info[genki_gene]
                if matching_gene['position'] == 'source':
                    gene_info['found_as_source'] += 1
                else:
                    gene_info['found_as_target'] += 1
                
                gene_info['total_connections'] += 1
                gene_info['directions'].add(connection['direction'])
                gene_info['clusters'].add(connection['cluster'])
                
                shap_val = connection['shap_value']
                gene_info['max_shap_value'] = max(gene_info['max_shap_value'], shap_val)
                gene_info['min_shap_value'] = min(gene_info['min_shap_value'], shap_val)
        
        # Calculate average SHAP values and find GenKI ranks
        genki_genes_list = self.genki_data['gene'].dropna().str.strip().tolist()
        
        for gene_name, gene_info in genki_gene_info.items():
            # Calculate average SHAP value
            total_shap = sum([conn['shap_value'] for conn in self.filtered_connections 
                             for mg in conn['matching_genki_genes'] 
                             if mg['genki_gene'] == gene_name])
            gene_info['avg_shap_value'] = total_shap / gene_info['total_connections'] if gene_info['total_connections'] > 0 else 0
            
            # Find GenKI rank
            try:
                gene_idx = next(i for i, g in enumerate(genki_genes_list) if g.strip().upper() == gene_name.upper())
                gene_info['genki_rank'] = gene_idx + 1
            except StopIteration:
                gene_info['genki_rank'] = None
            
            # Convert sets to lists for JSON serialization
            gene_info['directions'] = list(gene_info['directions'])
            gene_info['clusters'] = list(gene_info['clusters'])
            
            # Handle infinite min values
            if gene_info['min_shap_value'] == float('inf'):
                gene_info['min_shap_value'] = 0
        
        # Convert to list and sort by GenKI rank
        unique_genki = list(genki_gene_info.values())
        unique_genki.sort(key=lambda x: x['genki_rank'] if x['genki_rank'] is not None else float('inf'))
        
        self.unique_genki_genes = unique_genki
        
        logger.info(f"Found {len(unique_genki)} unique GenKI genes in SHAP connections")
        logger.info(f"  {len(unique_genki)}/{len(genki_genes_list)} GenKI genes ({len(unique_genki)/len(genki_genes_list)*100:.1f}%) have SHAP connections")
        
        return unique_genki
    
    def extract_unique_connected_molecules(self) -> List[Dict]:
        """Extract unique molecules/metabolites connected to GenKI genes"""
        logger.info("Extracting unique connected molecules from filtered connections...")
        
        connected_molecule_info = {}
        
        # Collect all connected molecules (non-GenKI genes in the connections)
        genki_genes_upper = [gene.upper() for gene in self.genki_data['gene'].dropna().str.strip().tolist()]
        
        for connection in self.filtered_connections:
            source = connection['source']
            target = connection['target']
            
            # Determine which molecule is the connected one (not the GenKI gene)
            connected_molecules = []
            
            # Check source
            if source.strip().upper() not in genki_genes_upper:
                connected_molecules.append({
                    'molecule': source,
                    'connection_type': 'connected_to_genki_target',
                    'genki_gene': target if target.strip().upper() in genki_genes_upper else None
                })
            
            # Check target  
            if target.strip().upper() not in genki_genes_upper:
                connected_molecules.append({
                    'molecule': target,
                    'connection_type': 'connected_to_genki_source',
                    'genki_gene': source if source.strip().upper() in genki_genes_upper else None
                })
            
            # Process each connected molecule
            for connected_mol in connected_molecules:
                if connected_mol['genki_gene'] is None:
                    continue  # Skip if we can't identify the GenKI gene
                
                molecule = connected_mol['molecule']
                molecule_key = molecule.strip().upper()
                
                if molecule_key not in connected_molecule_info:
                    connected_molecule_info[molecule_key] = {
                        'molecule_name': molecule.strip(),
                        'connected_to_genki_genes': set(),
                        'total_connections': 0,
                        'directions': set(),
                        'clusters': set(),
                        'connection_types': set(),
                        'avg_shap_value': 0,
                        'max_shap_value': 0,
                        'min_shap_value': float('inf'),
                        'is_flux_metabolite': self._is_flux_metabolite(molecule)
                    }
                
                mol_info = connected_molecule_info[molecule_key]
                mol_info['connected_to_genki_genes'].add(connected_mol['genki_gene'])
                mol_info['total_connections'] += 1
                mol_info['directions'].add(connection['direction'])
                mol_info['clusters'].add(connection['cluster'])
                mol_info['connection_types'].add(connected_mol['connection_type'])
                
                shap_val = connection['shap_value']
                mol_info['max_shap_value'] = max(mol_info['max_shap_value'], shap_val)
                mol_info['min_shap_value'] = min(mol_info['min_shap_value'], shap_val)
        
        # Calculate average SHAP values and convert sets to lists
        for molecule_key, mol_info in connected_molecule_info.items():
            # Calculate average SHAP value
            relevant_connections = []
            for conn in self.filtered_connections:
                if (conn['source'].strip().upper() == molecule_key or 
                    conn['target'].strip().upper() == molecule_key):
                    relevant_connections.append(conn['shap_value'])
            
            mol_info['avg_shap_value'] = sum(relevant_connections) / len(relevant_connections) if relevant_connections else 0
            
            # Convert sets to lists for JSON serialization
            mol_info['connected_to_genki_genes'] = list(mol_info['connected_to_genki_genes'])
            mol_info['directions'] = list(mol_info['directions'])
            mol_info['clusters'] = list(mol_info['clusters'])
            mol_info['connection_types'] = list(mol_info['connection_types'])
            mol_info['n_genki_connections'] = len(mol_info['connected_to_genki_genes'])
            
            # Handle infinite min values
            if mol_info['min_shap_value'] == float('inf'):
                mol_info['min_shap_value'] = 0
        
        # Convert to list and sort by total connections
        unique_connected = list(connected_molecule_info.values())
        unique_connected.sort(key=lambda x: x['total_connections'], reverse=True)
        
        self.unique_connected_molecules = unique_connected
        
        # Count flux metabolites
        flux_molecules = [mol for mol in unique_connected if mol['is_flux_metabolite']]
        
        logger.info(f"Found {len(unique_connected)} unique connected molecules")
        logger.info(f"  Flux metabolites: {len(flux_molecules)} ({len(flux_molecules)/len(unique_connected)*100:.1f}%)")
        logger.info(f"  Other molecules: {len(unique_connected) - len(flux_molecules)}")
        
        return unique_connected
    
    def create_comprehensive_lists(self) -> Dict:
        """Create comprehensive lists of all unique genes and metabolites"""
        logger.info("Creating comprehensive unique lists...")
        
        # Start with all 81 GenKI genes
        all_genki_genes = set()
        genki_genes_list = self.genki_data['gene'].dropna().str.strip().tolist()
        for gene in genki_genes_list:
            all_genki_genes.add(gene.upper())
        
        # Get connected genes (non-metabolite molecules that are genes)
        connected_genes = set()
        connected_metabolites = set()
        
        for mol in self.unique_connected_molecules:
            mol_name = mol['molecule_name'].strip()
            if mol['is_flux_metabolite']:
                # Clean up flux metabolite names
                if mol_name.startswith('FLUX:'):
                    clean_name = mol_name.replace('FLUX:', '').strip()
                else:
                    clean_name = mol_name
                connected_metabolites.add(clean_name.upper())
            else:
                # This is likely a gene/protein
                connected_genes.add(mol_name.upper())
        
        # Also get genes from SHAP connections that might not be in the connected_molecules
        # (genes that appear as both source and target in connections with GenKI genes)
        genki_genes_upper = [gene.upper() for gene in genki_genes_list]
        
        for connection in self.filtered_connections:
            source = connection['source'].strip().upper()
            target = connection['target'].strip().upper()
            
            # Add non-GenKI genes to connected genes
            if source not in genki_genes_upper and not self._is_flux_metabolite(source):
                connected_genes.add(source)
            if target not in genki_genes_upper and not self._is_flux_metabolite(target):
                connected_genes.add(target)
            
            # Add metabolites
            if source not in genki_genes_upper and self._is_flux_metabolite(source):
                clean_source = source.replace('FLUX:', '').strip()
                connected_metabolites.add(clean_source)
            if target not in genki_genes_upper and self._is_flux_metabolite(target):
                clean_target = target.replace('FLUX:', '').strip()
                connected_metabolites.add(clean_target)
        
        # Create comprehensive lists
        comprehensive_data = {
            'all_genki_genes': sorted(list(all_genki_genes)),
            'connected_genes': sorted(list(connected_genes)),
            'connected_metabolites': sorted(list(connected_metabolites)),
            'total_unique_genes': len(all_genki_genes) + len(connected_genes),
            'total_unique_metabolites': len(connected_metabolites),
            'genki_with_connections': len(self.unique_genki_genes),
            'genki_without_connections': len(all_genki_genes) - len(self.unique_genki_genes)
        }
        
        # Create combined unique gene list (GenKI + connected)
        all_unique_genes = set()
        all_unique_genes.update(all_genki_genes)
        all_unique_genes.update(connected_genes)
        comprehensive_data['all_unique_genes'] = sorted(list(all_unique_genes))
        
        logger.info(f"Comprehensive lists created:")
        logger.info(f"  All GenKI genes: {len(all_genki_genes)}")
        logger.info(f"  Connected genes: {len(connected_genes)}")
        logger.info(f"  Connected metabolites: {len(connected_metabolites)}")
        logger.info(f"  Total unique genes: {comprehensive_data['total_unique_genes']}")
        logger.info(f"  Total unique metabolites: {comprehensive_data['total_unique_metabolites']}")
        
        return comprehensive_data
    
    def _is_flux_metabolite(self, molecule_name: str) -> bool:
        """Determine if a molecule is likely a flux metabolite"""
        molecule = molecule_name.upper().strip()
        
        # Common metabolite indicators
        metabolite_indicators = [
            'FLUX:', 'AMP', 'ATP', 'ADP', 'GTP', 'GDP', 'CTP', 'CDP', 'UTP', 'UDP',
            'COA', 'COENZYME', 'GLUTAMATE', 'GLUTAMINE', 'GLYCINE', 'SERINE',
            'PYRUVATE', 'ACETYL', 'SUCCINATE', 'MALATE', 'CITRATE', 'OXALOACETATE',
            'GLUCOSE', 'FRUCTOSE', 'LACTATE', 'ASPARTATE', 'ARGININE', 'LYSINE',
            'LEUCINE', 'ISOLEUCINE', 'VALINE', 'METHIONINE', 'PHENYLALANINE',
            'TYROSINE', 'TRYPTOPHAN', 'THREONINE', 'HISTIDINE', 'PROLINE',
            'CYSTEINE', 'CHOLESTEROL', 'FATTY', 'LIPID'
        ]
        
        # Check if molecule name contains metabolite indicators
        for indicator in metabolite_indicators:
            if indicator in molecule:
                return True
        
        # Additional checks for metabolite-like patterns
        if (len(molecule) <= 6 and any(char.isdigit() for char in molecule)) or \
           ('_' in molecule and len(molecule.split('_')) <= 3) or \
           molecule.startswith('FLUX:'):
            return True
            
        return False
    
    def create_connection_summary(self) -> Dict:
        """Create summary statistics of connections"""
        logger.info("Creating connection summary...")
        
        summary = {
            'total_connections': len(self.filtered_connections),
            'unique_genki_genes': len(self.unique_genki_genes),
            'unique_connected_molecules': len(self.unique_connected_molecules),
            'flux_metabolites': len([mol for mol in self.unique_connected_molecules if mol['is_flux_metabolite']]),
            'non_flux_molecules': len([mol for mol in self.unique_connected_molecules if not mol['is_flux_metabolite']]),
            'directions': list(set([conn['direction'] for conn in self.filtered_connections])),
            'clusters': list(set([conn['cluster'] for conn in self.filtered_connections])),
            'genki_coverage_percent': len(self.unique_genki_genes) / len(self.genki_data) * 100 if len(self.genki_data) > 0 else 0
        }
        
        # Top connected GenKI genes
        summary['top_connected_genki_genes'] = sorted(
            self.unique_genki_genes, 
            key=lambda x: x['total_connections'], 
            reverse=True
        )[:10]
        
        # Top connected molecules
        summary['top_connected_molecules'] = sorted(
            self.unique_connected_molecules,
            key=lambda x: x['total_connections'],
            reverse=True
        )[:10]
        
        self.connection_summary = summary
        
        logger.info(f"Connection summary created:")
        logger.info(f"  Total connections: {summary['total_connections']}")
        logger.info(f"  Unique GenKI genes: {summary['unique_genki_genes']}")
        logger.info(f"  Unique connected molecules: {summary['unique_connected_molecules']}")
        logger.info(f"  Flux metabolites: {summary['flux_metabolites']}")
        logger.info(f"  GenKI coverage: {summary['genki_coverage_percent']:.1f}%")
        
        return summary
    
    def save_results(self, output_dir: str = None) -> Dict[str, str]:
        """Save analysis results to files"""
        if output_dir is None:
            output_dir = f"../Mongoose_post_hoc/flux_perturbmap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files_created = {}
        
        # Save filtered connections
        connections_file = output_path / "filtered_shap_connections.csv"
        if self.filtered_connections:
            # Flatten connections for CSV
            flattened_connections = []
            for conn in self.filtered_connections:
                base_conn = {
                    'cluster': conn['cluster'],
                    'direction': conn['direction'], 
                    'source': conn['source'],
                    'target': conn['target'],
                    'shap_value': conn['shap_value'],
                    'n_matching_genki_genes': len(conn['matching_genki_genes'])
                }
                
                # Add GenKI gene info
                genki_genes = [mg['genki_gene'] for mg in conn['matching_genki_genes']]
                positions = [mg['position'] for mg in conn['matching_genki_genes']]
                
                base_conn.update({
                    'genki_genes': '; '.join(genki_genes),
                    'genki_positions': '; '.join(positions)
                })
                
                flattened_connections.append(base_conn)
            
            conn_df = pd.DataFrame(flattened_connections)
            conn_df.to_csv(connections_file, index=False)
            files_created['connections'] = str(connections_file)
        
        # Save unique GenKI genes table
        genki_genes_file = output_path / "unique_genki_genes.csv"
        if self.unique_genki_genes:
            genki_df_data = []
            for gene in self.unique_genki_genes:
                gene_row = {
                    'gene_name': gene['gene_name'],
                    'genki_rank': gene['genki_rank'],
                    'total_connections': gene['total_connections'],
                    'found_as_source': gene['found_as_source'],
                    'found_as_target': gene['found_as_target'],
                    'avg_shap_value': gene['avg_shap_value'],
                    'max_shap_value': gene['max_shap_value'],
                    'min_shap_value': gene['min_shap_value'],
                    'directions': '; '.join(gene['directions']),
                    'clusters': '; '.join(map(str, gene['clusters'])),
                    'n_directions': len(gene['directions']),
                    'n_clusters': len(gene['clusters'])
                }
                
                # Add GenKI data columns
                if gene['genki_data']:
                    for key, value in gene['genki_data'].items():
                        if key != 'gene':  # Don't duplicate gene name
                            gene_row[f'genki_{key}'] = value
                
                genki_df_data.append(gene_row)
            
            genki_df = pd.DataFrame(genki_df_data)
            genki_df.to_csv(genki_genes_file, index=False)
            files_created['genki_genes'] = str(genki_genes_file)
        
        # Save unique connected molecules table
        molecules_file = output_path / "unique_connected_molecules.csv"
        if self.unique_connected_molecules:
            mol_df_data = []
            for mol in self.unique_connected_molecules:
                mol_row = {
                    'molecule_name': mol['molecule_name'],
                    'total_connections': mol['total_connections'],
                    'n_genki_connections': mol['n_genki_connections'],
                    'is_flux_metabolite': mol['is_flux_metabolite'],
                    'avg_shap_value': mol['avg_shap_value'],
                    'max_shap_value': mol['max_shap_value'],
                    'min_shap_value': mol['min_shap_value'],
                    'connected_genki_genes': '; '.join(mol['connected_to_genki_genes']),
                    'directions': '; '.join(mol['directions']),
                    'clusters': '; '.join(map(str, mol['clusters'])),
                    'connection_types': '; '.join(mol['connection_types']),
                    'n_directions': len(mol['directions']),
                    'n_clusters': len(mol['clusters'])
                }
                mol_df_data.append(mol_row)
            
            mol_df = pd.DataFrame(mol_df_data)
            mol_df.to_csv(molecules_file, index=False)
            files_created['molecules'] = str(molecules_file)
        
        # NEW: Save comprehensive lists
        # Save all unique genes list
        all_genes_file = output_path / "all_unique_genes.txt"
        with open(all_genes_file, 'w') as f:
            f.write(f"# All Unique Genes List (GenKI + Connected)\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total: {len(self.comprehensive_lists['all_unique_genes'])} genes\n")
            f.write(f"# GenKI genes: {len(self.comprehensive_lists['all_genki_genes'])}\n")
            f.write(f"# Connected genes: {len(self.comprehensive_lists['connected_genes'])}\n\n")
            for i, gene in enumerate(self.comprehensive_lists['all_unique_genes'], 1):
                f.write(f"{i}. {gene}\n")
        files_created['all_genes'] = str(all_genes_file)
        
        # Save GenKI genes list  
        genki_genes_list_file = output_path / "all_genki_genes.txt"
        with open(genki_genes_list_file, 'w') as f:
            f.write(f"# All 81 GenKI Genes\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total: {len(self.comprehensive_lists['all_genki_genes'])} genes\n")
            f.write(f"# With connections: {self.comprehensive_lists['genki_with_connections']}\n")
            f.write(f"# Without connections: {self.comprehensive_lists['genki_without_connections']}\n\n")
            for i, gene in enumerate(self.comprehensive_lists['all_genki_genes'], 1):
                f.write(f"{i}. {gene}\n")
        files_created['genki_genes_list'] = str(genki_genes_list_file)
        
        # Save connected genes list
        connected_genes_file = output_path / "connected_genes.txt" 
        with open(connected_genes_file, 'w') as f:
            f.write(f"# Connected Genes (Non-GenKI genes in SHAP connections)\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total: {len(self.comprehensive_lists['connected_genes'])} genes\n\n")
            for i, gene in enumerate(self.comprehensive_lists['connected_genes'], 1):
                f.write(f"{i}. {gene}\n")
        files_created['connected_genes'] = str(connected_genes_file)
        
        # Save metabolites list
        metabolites_file = output_path / "unique_metabolites.txt"
        with open(metabolites_file, 'w') as f:
            f.write(f"# Unique Metabolites from SHAP Connections\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n") 
            f.write(f"# Total: {len(self.comprehensive_lists['connected_metabolites'])} metabolites\n\n")
            for i, metabolite in enumerate(self.comprehensive_lists['connected_metabolites'], 1):
                f.write(f"{i}. {metabolite}\n")
        files_created['metabolites'] = str(metabolites_file)
        
        # Save comprehensive summary CSV
        comprehensive_csv = output_path / "comprehensive_summary.csv"
        comprehensive_df_data = []
        
        # Add GenKI genes
        for gene in self.comprehensive_lists['all_genki_genes']:
            comprehensive_df_data.append({
                'name': gene,
                'type': 'GenKI_gene',
                'has_shap_connections': gene.upper() in [g['gene_name'].upper() for g in self.unique_genki_genes],
                'category': 'gene'
            })
        
        # Add connected genes  
        for gene in self.comprehensive_lists['connected_genes']:
            comprehensive_df_data.append({
                'name': gene,
                'type': 'connected_gene', 
                'has_shap_connections': True,
                'category': 'gene'
            })
        
        # Add metabolites
        for metabolite in self.comprehensive_lists['connected_metabolites']:
            comprehensive_df_data.append({
                'name': metabolite,
                'type': 'metabolite',
                'has_shap_connections': True,
                'category': 'metabolite'
            })
        
        comprehensive_df = pd.DataFrame(comprehensive_df_data)
        comprehensive_df.to_csv(comprehensive_csv, index=False)
        files_created['comprehensive_summary'] = str(comprehensive_csv)
        
        # Save summary JSON
        summary_file = output_path / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.connection_summary, f, indent=2, default=str)
        files_created['summary'] = str(summary_file)
        
        # Save analysis report
        report_file = output_path / "analysis_report.md"
        report = self._generate_report()
        with open(report_file, 'w') as f:
            f.write(report)
        files_created['report'] = str(report_file)
        
        logger.info(f"Results saved to {output_path}/")
        for file_type, file_path in files_created.items():
            logger.info(f"  {file_type}: {Path(file_path).name}")
        
        return files_created
    
    def _generate_report(self) -> str:
        """Generate markdown analysis report"""
        report = f"""# Flux PerturbMap Connection Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Summary

### Input Data
- **SHAP File**: `{self.shap_file.name}`
- **GenKI File**: `{self.genki_file.name}`
- **Original SHAP entries**: {len(self.shap_data):,}
- **GenKI genes**: {len(self.genki_data):,}

### Key Findings
- **Filtered connections**: {self.connection_summary['total_connections']:,} ({self.connection_summary['total_connections']/len(self.shap_data)*100:.1f}% of total SHAP entries)
- **Unique GenKI genes with connections**: {self.connection_summary['unique_genki_genes']:,} ({self.connection_summary['genki_coverage_percent']:.1f}% of GenKI genes)
- **Unique connected molecules**: {self.connection_summary['unique_connected_molecules']:,}
  - Flux metabolites: {self.connection_summary['flux_metabolites']:,}
  - Other molecules: {self.connection_summary['non_flux_molecules']:,}

### Directions Analyzed
{chr(10).join([f'- {direction}' for direction in self.connection_summary['directions']])}

### Clusters Found
- Clusters: {', '.join(map(str, sorted(self.connection_summary['clusters'])))}

---

## Top Connected GenKI Genes

| Rank | Gene | Connections | As Source | As Target | Avg SHAP | Directions | Clusters |
|------|------|-------------|-----------|-----------|----------|------------|----------|"""

        for i, gene in enumerate(self.connection_summary['top_connected_genki_genes'], 1):
            report += f"""
| {i} | {gene['gene_name']} | {gene['total_connections']} | {gene['found_as_source']} | {gene['found_as_target']} | {gene['avg_shap_value']:.4f} | {len(gene['directions'])} | {len(gene['clusters'])} |"""
        
        report += f"""

---

## Top Connected Molecules

| Rank | Molecule | Connections | GenKI Genes | Type | Avg SHAP | Directions | Clusters |
|------|----------|-------------|-------------|------|----------|------------|----------|"""

        for i, mol in enumerate(self.connection_summary['top_connected_molecules'], 1):
            mol_type = "Flux" if mol['is_flux_metabolite'] else "Other"
            report += f"""
| {i} | {mol['molecule_name']} | {mol['total_connections']} | {mol['n_genki_connections']} | {mol_type} | {mol['avg_shap_value']:.4f} | {len(mol['directions'])} | {len(mol['clusters'])} |"""
        
        report += f"""

---

## Analysis Details

### Connection Distribution by Position
"""
        
        # Calculate position statistics
        source_connections = len([conn for conn in self.filtered_connections if 'source' in conn['genki_positions']])
        target_connections = len([conn for conn in self.filtered_connections if 'target' in conn['genki_positions']])
        both_connections = len([conn for conn in self.filtered_connections if len(conn['genki_positions']) > 1])
        
        report += f"""
- GenKI genes found as **source**: {source_connections} connections
- GenKI genes found as **target**: {target_connections} connections  
- GenKI genes found in **both positions**: {both_connections} connections

### Flux Metabolite Analysis
- **{self.connection_summary['flux_metabolites']}** unique flux metabolites identified
- **{self.connection_summary['non_flux_molecules']}** other molecules (genes, proteins, etc.)
- Flux metabolites represent **{self.connection_summary['flux_metabolites']/self.connection_summary['unique_connected_molecules']*100:.1f}%** of all connected molecules

---

*Analysis completed using Flux PerturbMap Connection Analyzer*
*Results focus on GenKI gene connections without ground truth validation*
"""
        
        return report
    
    def run_analysis(self, output_dir: str = None) -> bool:
        """Run complete analysis pipeline"""
        logger.info("Starting flux PerturbMap connection analysis...")
        
        try:
            # Load data
            if not self.load_data():
                return False
            
            # Run analysis steps
            self.filter_shap_connections()
            self.extract_unique_genki_genes()
            self.extract_unique_connected_molecules()
            self.comprehensive_lists = self.create_comprehensive_lists()  # NEW
            self.create_connection_summary()
            
            # Save results
            files_created = self.save_results(output_dir)
            
            # Print summary to console
            self._print_summary()
            
            logger.info("Flux PerturbMap connection analysis completed successfully!")
            logger.info(f"Results saved to: {list(files_created.values())[0].split('/')[0]}/")
            
            return True
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_summary(self):
        """Print analysis summary to console"""
        print("\n" + "="*80)
        print("FLUX PERTURBMAP CONNECTION ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"Input Data:")
        print(f"   SHAP entries: {len(self.shap_data):,}")
        print(f"   GenKI genes: {len(self.genki_data):,}")
        
        print(f"\nFiltered Connections:")
        print(f"   Total connections with GenKI genes: {len(self.filtered_connections):,}")
        print(f"   Coverage: {len(self.filtered_connections)/len(self.shap_data)*100:.1f}% of all SHAP entries")
        
        # Position breakdown
        source_connections = len([conn for conn in self.filtered_connections if 'source' in conn['genki_positions']])
        target_connections = len([conn for conn in self.filtered_connections if 'target' in conn['genki_positions']])
        both_connections = len([conn for conn in self.filtered_connections if len(conn['genki_positions']) > 1])
        
        print(f"   - GenKI as source: {source_connections}")
        print(f"   - GenKI as target: {target_connections}")
        print(f"   - GenKI in both: {both_connections}")
        
        print(f"\nUnique GenKI Genes:")
        print(f"   Genes with SHAP connections: {len(self.unique_genki_genes)}")
        print(f"   Coverage: {len(self.unique_genki_genes)}/{len(self.genki_data)} ({self.connection_summary['genki_coverage_percent']:.1f}% of GenKI genes)")
        
        if self.unique_genki_genes:
            top_3_genki = sorted(self.unique_genki_genes, key=lambda x: x['total_connections'], reverse=True)[:3]
            print(f"   Top connected GenKI genes:")
            for i, gene in enumerate(top_3_genki, 1):
                print(f"   {i}. {gene['gene_name']} ({gene['total_connections']} connections)")
        
        print(f"\nUnique Connected Molecules:")
        print(f"   Total molecules: {len(self.unique_connected_molecules)}")
        print(f"   - Flux metabolites: {self.connection_summary['flux_metabolites']} ({self.connection_summary['flux_metabolites']/len(self.unique_connected_molecules)*100:.1f}%)")
        print(f"   - Other molecules: {self.connection_summary['non_flux_molecules']} ({self.connection_summary['non_flux_molecules']/len(self.unique_connected_molecules)*100:.1f}%)")
        
        if self.unique_connected_molecules:
            top_3_molecules = sorted(self.unique_connected_molecules, key=lambda x: x['total_connections'], reverse=True)[:3]
            print(f"   Top connected molecules:")
            for i, mol in enumerate(top_3_molecules, 1):
                mol_type = "Flux" if mol['is_flux_metabolite'] else "Other"
                print(f"   {i}. {mol['molecule_name']} ({mol['total_connections']} connections, {mol_type})")
        
        print(f"\nComprehensive Lists:")
        print(f"   All GenKI genes: {len(self.comprehensive_lists['all_genki_genes'])}")
        print(f"   - With SHAP connections: {self.comprehensive_lists['genki_with_connections']}")
        print(f"   - Without SHAP connections: {self.comprehensive_lists['genki_without_connections']}")
        print(f"   Connected genes: {len(self.comprehensive_lists['connected_genes'])}")
        print(f"   Unique metabolites: {len(self.comprehensive_lists['connected_metabolites'])}")
        print(f"   TOTAL unique genes: {self.comprehensive_lists['total_unique_genes']}")
        print(f"   TOTAL unique metabolites: {self.comprehensive_lists['total_unique_metabolites']}")
        
        print(f"\nDirections & Clusters:")
        print(f"   Directions: {', '.join(self.connection_summary['directions'])}")
        print(f"   Clusters: {', '.join(map(str, sorted(self.connection_summary['clusters'])))}")
        
        print("="*80)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Flux PerturbMap Connection Analyzer - Identify SHAP connections containing GenKI genes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python flux_perturbmap_analyzer.py --shap_file feature_feature_importance_flux_3modalities.csv --genki_file genki_results.csv
  
  # With custom output directory
  python flux_perturbmap_analyzer.py --shap_file shap.csv --genki_file genki.csv --output_dir my_analysis
        """
    )
    
    # Required arguments
    parser.add_argument('--shap_file', type=str, default=None,
                       help='Path to SHAP feature importance CSV file (auto-detected if not provided)')
    parser.add_argument('--cross_modal_base_path', type=str, default='../Task2_CMP',
                       help='Base path to search for cross-modal analysis directories')
    parser.add_argument('--genki_file', type=str, required=True, 
                       help='Path to GenKI digital knockout results CSV file')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory name (default: ../Mongoose_post_hoc/flux_perturbmap_analysis_TIMESTAMP)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find or use SHAP file
    shap_file_path = None
    if args.shap_file and Path(args.shap_file).exists():
        shap_file_path = args.shap_file
        print(f"Using specified SHAP file: {shap_file_path}")
    else:
        if args.shap_file:
            print(f"Specified SHAP file not found: {args.shap_file}")
        print("Searching for latest cross-modal analysis directory...")
        latest_dir = find_latest_cross_modal_dir(base_path=args.cross_modal_base_path)
        if not latest_dir:
            logger.error(f"No cross-modal directories found in {args.cross_modal_base_path}")
            return 1
        
        # Look for the feature importance file in the latest directory
        feature_file = Path(latest_dir) / "feature_feature_importance_flux_3modalities.csv"
        if feature_file.exists():
            shap_file_path = str(feature_file)
            print(f"Using auto-detected SHAP file: {shap_file_path}")
        else:
            logger.error(f"Feature importance file not found in {latest_dir}")
            logger.error(f"Expected: {feature_file}")
            return 1
    
    # Validate GenKI file
    if not Path(args.genki_file).exists():
        logger.error(f"GenKI file not found: {args.genki_file}")
        return 1
    
    # Run analysis
    analyzer = FluxPerturbMapAnalyzer(
        shap_file=shap_file_path,
        genki_file=args.genki_file
    )
    
    success = analyzer.run_analysis(output_dir=args.output_dir)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())