#!/usr/bin/env python3
"""
Enhanced PerturbMap Analysis Pipeline
=====================================

Comprehensive analysis tool for comparing GenKI digital knockout results,
UnitedNet SHAP feature importance, and ground truth DEGs.

NEW FEATURE: Outputs final gene list including GenKI genes + significant SHAP connection genes
with p-value < 0.05 in DEGs.

Features:
- Checks BOTH Source AND Target columns for gene matching
- Creates timestamped output directories
- Generates detailed reports and CSV files
- NEW: Identifies significant SHAP connection genes in DEGs (p < 0.05)
- NEW: Outputs final combined gene list
- Provides console output and file exports

Usage:
    python enhanced_perturbmap_analyzer.py --shap_file path/to/shap.csv --genki_file path/to/genki.csv --deg_file path/to/degs.csv

Author: Enhanced for PerturbMap analysis with significant connections
Date: 2025
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Set
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPerturbMapAnalyzer:
    """Enhanced PerturbMap analysis pipeline with significant connection gene identification"""
    
    def __init__(self, shap_file: str, genki_file: str, deg_file: str, top_n: int = 5, pvalue_threshold: float = 0.05):
        self.shap_file = Path(shap_file)
        self.genki_file = Path(genki_file)
        self.deg_file = Path(deg_file)
        self.top_n = top_n
        self.pvalue_threshold = pvalue_threshold
        
        # Data containers
        self.shap_data = None
        self.genki_data = None
        self.deg_data = None
        
        # Analysis results
        self.matched_genes = []
        self.unique_molecules = []
        self.deg_overlap = {}
        self.shap_connections = {}
        self.significant_connection_genes = []  # NEW: genes from SHAP connections with p < 0.05 in DEGs
        self.final_gene_list = []  # NEW: combined final gene list
        
        # Report data
        self.report_data = {}
        
        logger.info(f"Initialized Enhanced PerturbMapAnalyzer with top_n={top_n}, p-value threshold={pvalue_threshold}")
    
    def load_data(self) -> bool:
        """Load all input files and validate data"""
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
            # Handle the unnamed first column (gene names)
            if self.genki_data.columns[0] == 'Unnamed: 0' or self.genki_data.columns[0] == '':
                self.genki_data = self.genki_data.rename(columns={self.genki_data.columns[0]: 'gene'})
            
            # Load DEG data
            logger.info(f"Loading DEG data from {self.deg_file}")
            self.deg_data = pd.read_csv(self.deg_file)
            required_deg_cols = ['names', 'logfoldchanges', 'pvals', 'pvals_adj']
            if not all(col in self.deg_data.columns for col in required_deg_cols):
                raise ValueError(f"DEG file missing required columns: {required_deg_cols}")
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"  SHAP: {len(self.shap_data)} entries")
            logger.info(f"  GenKI: {len(self.genki_data)} genes") 
            logger.info(f"  DEGs: {len(self.deg_data)} genes")
            logger.info(f"  Significant DEGs (p < {self.pvalue_threshold}): {len(self.deg_data[self.deg_data['pvals'] < self.pvalue_threshold])}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def find_matched_genes(self) -> List[Dict]:
        """Find genes that appear in GenKI and SHAP sources OR targets"""
        logger.info("Finding matched genes between GenKI and SHAP sources/targets...")
        
        # Extract gene lists
        genki_genes = self.genki_data['gene'].dropna().str.strip().tolist()
        shap_sources = self.shap_data['Source'].dropna().str.strip().unique().tolist()
        shap_targets = self.shap_data['Target'].dropna().str.strip().unique().tolist()
        
        # Combine sources and targets for comprehensive matching
        all_shap_genes = list(set(shap_sources + shap_targets))
        
        logger.info(f"  GenKI genes: {len(genki_genes)}")
        logger.info(f"  SHAP sources: {len(shap_sources)}")
        logger.info(f"  SHAP targets: {len(shap_targets)}")
        logger.info(f"  Combined SHAP genes (sources + targets): {len(all_shap_genes)}")
        
        # Case-insensitive matching
        genki_upper = [g.upper() for g in genki_genes]
        shap_upper = [s.upper() for s in all_shap_genes]
        
        matched = []
        for i, genki_gene in enumerate(genki_genes):
            genki_upper_gene = genki_gene.upper()
            if genki_upper_gene in shap_upper:
                shap_idx = shap_upper.index(genki_upper_gene)
                matched_shap_gene = all_shap_genes[shap_idx]
                
                # Determine if found in source, target, or both
                found_in = []
                if matched_shap_gene in shap_sources:
                    found_in.append('source')
                if matched_shap_gene in shap_targets:
                    found_in.append('target')
                
                matched.append({
                    'genki': genki_gene,
                    'shap': matched_shap_gene,
                    'genki_rank': i + 1,
                    'found_in': ' & '.join(found_in),
                    'genki_data': self.genki_data.iloc[i].to_dict()
                })
        
        self.matched_genes = matched
        logger.info(f"Found {len(matched)} matched genes ({len(matched)/len(genki_genes)*100:.1f}% of GenKI genes)")
        
        # Log breakdown by location
        source_matches = [m for m in matched if 'source' in m['found_in']]
        target_matches = [m for m in matched if 'target' in m['found_in']]
        both_matches = [m for m in matched if 'source' in m['found_in'] and 'target' in m['found_in']]
        
        logger.info(f"  Found in sources: {len(source_matches)}")
        logger.info(f"  Found in targets: {len(target_matches)}")
        logger.info(f"  Found in both: {len(both_matches)}")
        
        return matched
    
    def extract_shap_connections(self) -> Dict:
        """Extract top SHAP connections for matched genes"""
        logger.info(f"Extracting top {self.top_n} SHAP connections for each matched gene...")
        
        connections = {}
        directions = self.shap_data['Direction'].unique()
        
        for gene_info in self.matched_genes:
            gene_name = gene_info['shap']  # Use SHAP format name
            connections[gene_name] = {}
            
            for direction in directions:
                # Filter connections for this gene and direction
                gene_connections = pd.DataFrame()
                
                # Look for gene as source
                if 'source' in gene_info['found_in']:
                    source_connections = self.shap_data[
                        (self.shap_data['Source'] == gene_name) & 
                        (self.shap_data['Direction'] == direction)
                    ].copy()
                    gene_connections = pd.concat([gene_connections, source_connections], ignore_index=True)
                
                # Look for gene as target
                if 'target' in gene_info['found_in']:
                    target_connections = self.shap_data[
                        (self.shap_data['Target'] == gene_name) & 
                        (self.shap_data['Direction'] == direction)
                    ].copy()
                    gene_connections = pd.concat([gene_connections, target_connections], ignore_index=True)
                
                if len(gene_connections) == 0:
                    connections[gene_name][direction] = []
                    continue
                
                # Remove duplicates
                gene_connections = gene_connections.drop_duplicates()
                
                # Sort by SHAP value (descending)
                gene_connections = gene_connections.sort_values('Value', ascending=False)
                
                # Get top N connections
                top_connections = []
                for idx, row in gene_connections.head(self.top_n).iterrows():
                    connected_gene = row['Target'] if row['Source'] == gene_name else row['Source']
                    top_connections.append({
                        'connected_gene': connected_gene,
                        'shap_value': row['Value'],
                        'cluster': row['Cluster'],
                        'connection_type': 'as_source' if row['Source'] == gene_name else 'as_target'
                    })
                
                connections[gene_name][direction] = top_connections
        
        self.shap_connections = connections
        return connections
    
    def identify_significant_connection_genes(self) -> List[Dict]:
        """NEW: Identify unique SHAP connection genes that have p-value < threshold in DEGs"""
        logger.info(f"Identifying significant connection genes (p < {self.pvalue_threshold})...")
        
        # Get all unique connection genes from SHAP connections
        connection_genes = set()
        for gene_connections in self.shap_connections.values():
            for direction_connections in gene_connections.values():
                for conn in direction_connections:
                    connected_gene = conn['connected_gene']
                    # Clean gene name (remove _SIG suffix if present)
                    if connected_gene.endswith('_SIG'):
                        connected_gene = connected_gene.replace('_SIG', '')
                    connection_genes.add(connected_gene.upper())
        
        logger.info(f"  Total unique connection genes: {len(connection_genes)}")
        
        # Filter DEGs by p-value threshold
        significant_degs = self.deg_data[self.deg_data['pvals'] < self.pvalue_threshold].copy()
        deg_names_upper = significant_degs['names'].str.strip().str.upper().tolist()
        
        logger.info(f"  Significant DEGs (p < {self.pvalue_threshold}): {len(significant_degs)}")
        
        # Find matches
        significant_connections = []
        for conn_gene in connection_genes:
            if conn_gene in deg_names_upper:
                deg_idx = deg_names_upper.index(conn_gene)
                deg_row = significant_degs.iloc[deg_idx]
                
                # Find which GenKI genes this connection is related to
                related_genki_genes = []
                for genki_gene, gene_connections in self.shap_connections.items():
                    for direction, connections in gene_connections.items():
                        for conn in connections:
                            clean_conn_gene = conn['connected_gene']
                            if clean_conn_gene.endswith('_SIG'):
                                clean_conn_gene = clean_conn_gene.replace('_SIG', '')
                            if clean_conn_gene.upper() == conn_gene:
                                related_genki_genes.append({
                                    'genki_gene': genki_gene,
                                    'direction': direction,
                                    'shap_value': conn['shap_value'],
                                    'connection_type': conn['connection_type']
                                })
                
                significant_connections.append({
                    'connection_gene': conn_gene,
                    'original_deg_name': deg_row['names'],
                    'logfc': deg_row['logfoldchanges'],
                    'pval': deg_row['pvals'],
                    'pval_adj': deg_row['pvals_adj'],
                    'score': deg_row.get('scores', 0),
                    'related_genki_genes': related_genki_genes,
                    'n_genki_connections': len(related_genki_genes)
                })
        
        # Sort by p-value (most significant first)
        significant_connections = sorted(significant_connections, key=lambda x: x['pval'])
        
        self.significant_connection_genes = significant_connections
        
        logger.info(f"Found {len(significant_connections)} significant connection genes")
        logger.info(f"  Connection genes in significant DEGs: {len(significant_connections)}/{len(connection_genes)} ({len(significant_connections)/len(connection_genes)*100:.1f}%)")
        
        return significant_connections
    
    def create_final_gene_list(self) -> List[Dict]:
        """NEW: Create final combined gene list: GenKI genes + significant SHAP connection genes"""
        logger.info("Creating final combined gene list...")
        
        final_genes = []
        
        # Add all GenKI genes first
        genki_genes_upper = []
        for i, row in self.genki_data.iterrows():
            gene_name = row['gene']
            if pd.notna(gene_name):
                gene_name = gene_name.strip()
                genki_genes_upper.append(gene_name.upper())
                
                # Check if this gene is also in DEGs
                deg_info = None
                deg_names_upper = self.deg_data['names'].str.strip().str.upper().tolist()
                if gene_name.upper() in deg_names_upper:
                    deg_idx = deg_names_upper.index(gene_name.upper())
                    deg_row = self.deg_data.iloc[deg_idx]
                    deg_info = {
                        'logfc': deg_row['logfoldchanges'],
                        'pval': deg_row['pvals'],
                        'pval_adj': deg_row['pvals_adj'],
                        'is_significant': deg_row['pvals'] < self.pvalue_threshold
                    }
                
                final_genes.append({
                    'gene_name': gene_name,
                    'source': 'GenKI',
                    'genki_rank': i + 1,
                    'genki_data': row.to_dict(),
                    'deg_info': deg_info,
                    'is_in_degs': deg_info is not None,
                    'is_significant_deg': deg_info is not None and deg_info['is_significant']
                })
        
        # Add significant connection genes that are not already in GenKI
        for conn in self.significant_connection_genes:
            conn_gene = conn['connection_gene']
            if conn_gene not in genki_genes_upper:  # Only add if not already in GenKI
                final_genes.append({
                    'gene_name': conn['original_deg_name'],  # Use original DEG name
                    'source': 'SHAP_Connection',
                    'genki_rank': None,
                    'genki_data': None,
                    'deg_info': {
                        'logfc': conn['logfc'],
                        'pval': conn['pval'],
                        'pval_adj': conn['pval_adj'],
                        'is_significant': True  # By definition, since these are significant connections
                    },
                    'is_in_degs': True,
                    'is_significant_deg': True,
                    'connection_info': {
                        'related_genki_genes': conn['related_genki_genes'],
                        'n_genki_connections': conn['n_genki_connections']
                    }
                })
        
        self.final_gene_list = final_genes
        
        # Calculate statistics
        genki_count = len([g for g in final_genes if g['source'] == 'GenKI'])
        connection_count = len([g for g in final_genes if g['source'] == 'SHAP_Connection'])
        total_in_degs = len([g for g in final_genes if g['is_in_degs']])
        total_significant = len([g for g in final_genes if g['is_significant_deg']])
        
        logger.info(f"Final gene list created:")
        logger.info(f"  Total genes: {len(final_genes)}")
        logger.info(f"  GenKI genes: {genki_count}")
        logger.info(f"  SHAP connection genes: {connection_count}")
        logger.info(f"  Found in DEGs: {total_in_degs} ({total_in_degs/len(final_genes)*100:.1f}%)")
        logger.info(f"  Significant DEGs (p < {self.pvalue_threshold}): {total_significant} ({total_significant/len(final_genes)*100:.1f}%)")
        
        return final_genes
    
    def create_unique_molecules_list(self) -> List[str]:
        """Create unique list of molecules from GenKI genes and SHAP data"""
        logger.info("Creating unique molecules list...")
        
        # Get all GenKI genes
        genki_genes = self.genki_data['gene'].dropna().str.strip().tolist()
        
        # Get all unique SHAP sources and targets
        all_shap_sources = self.shap_data['Source'].dropna().str.strip().unique().tolist()
        all_shap_targets = self.shap_data['Target'].dropna().str.strip().unique().tolist()
        all_shap_genes = list(set(all_shap_sources + all_shap_targets))
        
        # Get connected genes from SHAP connections
        connected_genes = []
        for gene_connections in self.shap_connections.values():
            for direction_connections in gene_connections.values():
                for conn in direction_connections:
                    connected_gene = conn['connected_gene']
                    # Remove _SIG suffix if present
                    if connected_gene.endswith('_SIG'):
                        connected_gene = connected_gene.replace('_SIG', '')
                    connected_genes.append(connected_gene)
        
        # Combine all molecules and deduplicate (case-insensitive)
        all_molecules = []
        seen_upper = set()
        
        for molecule in genki_genes + all_shap_genes + connected_genes:
            molecule_upper = molecule.upper()
            if molecule_upper not in seen_upper:
                all_molecules.append(molecule.upper())
                seen_upper.add(molecule_upper)
        
        self.unique_molecules = sorted(all_molecules)
        logger.info(f"Created unique molecules list: {len(self.unique_molecules)} molecules")
        
        return self.unique_molecules
    
    def analyze_deg_overlap(self) -> Dict:
        """Analyze overlap between unique molecules and ground truth DEGs"""
        logger.info("Analyzing DEG overlap...")
        
        # Get DEG names (case-insensitive)
        deg_names = self.deg_data['names'].dropna().str.strip().str.upper().tolist()
        deg_names_original = self.deg_data['names'].dropna().str.strip().tolist()
        
        # Find matches
        matches = []
        for molecule in self.unique_molecules:
            if molecule in deg_names:
                deg_idx = deg_names.index(molecule)
                deg_row = self.deg_data.iloc[deg_idx]
                matches.append({
                    'molecule': molecule,
                    'original_name': deg_names_original[deg_idx],
                    'logfc': deg_row['logfoldchanges'],
                    'pval': deg_row['pvals'],
                    'pval_adj': deg_row['pvals_adj'],
                    'score': deg_row.get('scores', 0),
                    'is_significant': deg_row['pvals'] < self.pvalue_threshold
                })
        
        # Sort by absolute log fold change
        matches = sorted(matches, key=lambda x: abs(x['logfc']), reverse=True)
        
        # Categorize by source
        genki_genes_upper = [g.upper() for g in self.genki_data['gene'].dropna().str.strip().tolist()]
        genki_matches = [m for m in matches if m['molecule'] in genki_genes_upper]
        shap_matches = [m for m in matches if m['molecule'] not in genki_genes_upper]
        
        # Calculate statistics
        overlap_stats = {
            'total_molecules': len(self.unique_molecules),
            'total_matches': len(matches),
            'significant_matches': len([m for m in matches if m['is_significant']]),
            'coverage_percent': len(matches) / len(self.unique_molecules) * 100 if len(self.unique_molecules) > 0 else 0,
            'significant_coverage_percent': len([m for m in matches if m['is_significant']]) / len(self.unique_molecules) * 100 if len(self.unique_molecules) > 0 else 0,
            'genki_matches': len(genki_matches),
            'genki_significant_matches': len([m for m in genki_matches if m['is_significant']]),
            'genki_total': len(genki_genes_upper),
            'genki_percent': len(genki_matches) / len(genki_genes_upper) * 100 if len(genki_genes_upper) > 0 else 0,
            'genki_significant_percent': len([m for m in genki_matches if m['is_significant']]) / len(genki_genes_upper) * 100 if len(genki_genes_upper) > 0 else 0,
            'shap_matches': len(shap_matches),
            'shap_significant_matches': len([m for m in shap_matches if m['is_significant']]),
            'matches': matches,
            'genki_matched_list': genki_matches,
            'shap_matched_list': shap_matches
        }
        
        self.deg_overlap = overlap_stats
        logger.info(f"DEG overlap analysis complete:")
        logger.info(f"  Total coverage: {overlap_stats['coverage_percent']:.1f}% ({overlap_stats['total_matches']}/{overlap_stats['total_molecules']})")
        logger.info(f"  Significant coverage: {overlap_stats['significant_coverage_percent']:.1f}% ({overlap_stats['significant_matches']}/{overlap_stats['total_molecules']})")
        logger.info(f"  GenKI coverage: {overlap_stats['genki_percent']:.1f}% ({overlap_stats['genki_matches']}/{overlap_stats['genki_total']})")
        logger.info(f"  GenKI significant: {overlap_stats['genki_significant_percent']:.1f}% ({overlap_stats['genki_significant_matches']}/{overlap_stats['genki_total']})")
        logger.info(f"  SHAP coverage: {overlap_stats['shap_matches']} matches ({overlap_stats['shap_significant_matches']} significant)")
        
        return overlap_stats
    
    def generate_report(self) -> str:
        """Generate comprehensive markdown report"""
        logger.info("Generating comprehensive report...")
        
        # Prepare report data
        self.report_data = {
            'analysis_params': {
                'shap_file': str(self.shap_file),
                'genki_file': str(self.genki_file),
                'deg_file': str(self.deg_file),
                'top_n_connections': self.top_n,
                'pvalue_threshold': self.pvalue_threshold,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'version': 'Enhanced version - includes significant connection genes analysis'
            },
            'data_summary': {
                'shap_entries': len(self.shap_data),
                'genki_genes': len(self.genki_data),
                'deg_genes': len(self.deg_data),
                'significant_degs': len(self.deg_data[self.deg_data['pvals'] < self.pvalue_threshold]),
                'matched_genes': len(self.matched_genes),
                'unique_molecules': len(self.unique_molecules),
                'significant_connection_genes': len(self.significant_connection_genes),
                'final_gene_list_size': len(self.final_gene_list)
            },
            'matched_genes': self.matched_genes,
            'shap_connections': self.shap_connections,
            'deg_overlap': self.deg_overlap,
            'significant_connection_genes': self.significant_connection_genes,
            'final_gene_list': self.final_gene_list
        }
        
        # Generate markdown report
        report = f"""# Enhanced PerturbMap Analysis Report

Generated: {self.report_data['analysis_params']['analysis_date']}

## Analysis Parameters
- **Version**: {self.report_data['analysis_params']['version']}
- **SHAP File**: `{self.report_data['analysis_params']['shap_file']}`
- **GenKI File**: `{self.report_data['analysis_params']['genki_file']}`
- **DEG File**: `{self.report_data['analysis_params']['deg_file']}`
- **Top N Connections**: {self.report_data['analysis_params']['top_n_connections']}
- **P-value Threshold**: {self.report_data['analysis_params']['pvalue_threshold']}

## Data Summary
- **SHAP Entries**: {self.report_data['data_summary']['shap_entries']:,}
- **GenKI Genes**: {self.report_data['data_summary']['genki_genes']:,}
- **Ground Truth DEGs**: {self.report_data['data_summary']['deg_genes']:,}
- **Significant DEGs (p < {self.pvalue_threshold})**: {self.report_data['data_summary']['significant_degs']:,}
- **Matched Genes**: {self.report_data['data_summary']['matched_genes']:,}
- **Unique Molecules**: {self.report_data['data_summary']['unique_molecules']:,}
- **Significant Connection Genes**: {self.report_data['data_summary']['significant_connection_genes']:,}
- **Final Gene List**: {self.report_data['data_summary']['final_gene_list_size']:,}

---

## Key Findings

### Gene Matching Results
- **{len(self.matched_genes)} genes** found in GenKI predictions and SHAP sources/targets
- **{len(self.matched_genes)/len(self.genki_data)*100:.1f}%** of GenKI genes have SHAP connections

**Matching Breakdown:**
"""
        
        # Add breakdown by location
        source_matches = [m for m in self.matched_genes if 'source' in m['found_in']]
        target_matches = [m for m in self.matched_genes if 'target' in m['found_in']]
        both_matches = [m for m in self.matched_genes if 'source' in m['found_in'] and 'target' in m['found_in']]
        
        report += f"""
- Found in SHAP Sources: {len(source_matches)} genes
- Found in SHAP Targets: {len(target_matches)} genes  
- Found in Both: {len(both_matches)} genes

### Ground Truth Validation
- **{self.deg_overlap['total_matches']}/{self.deg_overlap['total_molecules']} molecules ({self.deg_overlap['coverage_percent']:.1f}%)** found in actual DEGs
- **{self.deg_overlap['significant_matches']}/{self.deg_overlap['total_molecules']} molecules ({self.deg_overlap['significant_coverage_percent']:.1f}%)** found in significant DEGs (p < {self.pvalue_threshold})
- **GenKI Performance**: {self.deg_overlap['genki_matches']}/{self.deg_overlap['genki_total']} ({self.deg_overlap['genki_percent']:.1f}%) validation rate
- **GenKI Significant Performance**: {self.deg_overlap['genki_significant_matches']}/{self.deg_overlap['genki_total']} ({self.deg_overlap['genki_significant_percent']:.1f}%) significant validation rate
- **SHAP Performance**: {self.deg_overlap['shap_matches']} targets validated ({self.deg_overlap['shap_significant_matches']} significant)

### NEW: Significant Connection Genes Analysis
- **{len(self.significant_connection_genes)} connection genes** found in significant DEGs (p < {self.pvalue_threshold})
- These represent SHAP-identified connections that are biologically validated as differentially expressed

### NEW: Final Combined Gene List
- **{len(self.final_gene_list)} total genes** in final list
- **GenKI genes**: {len([g for g in self.final_gene_list if g['source'] == 'GenKI'])}
- **Additional significant connection genes**: {len([g for g in self.final_gene_list if g['source'] == 'SHAP_Connection'])}
- **Found in DEGs**: {len([g for g in self.final_gene_list if g['is_in_degs']])} ({len([g for g in self.final_gene_list if g['is_in_degs']])/len(self.final_gene_list)*100:.1f}%)
- **Significant DEGs**: {len([g for g in self.final_gene_list if g['is_significant_deg']])} ({len([g for g in self.final_gene_list if g['is_significant_deg']])/len(self.final_gene_list)*100:.1f}%)

---

## Significant Connection Genes Details

"""
        
        # Add significant connection genes section
        if self.significant_connection_genes:
            report += f"**Top {min(20, len(self.significant_connection_genes))} Most Significant Connection Genes:**\n\n"
            for i, conn in enumerate(self.significant_connection_genes[:20], 1):
                direction = "UP" if conn['logfc'] > 0 else "DOWN"
                genki_connections = ", ".join([f"{rel['genki_gene']} ({rel['direction']})" for rel in conn['related_genki_genes'][:3]])
                if len(conn['related_genki_genes']) > 3:
                    genki_connections += f" +{len(conn['related_genki_genes'])-3} more"
                
                report += f"{i}. **{conn['connection_gene']}** ({direction}) - p: {conn['pval']:.2e}, logFC: {conn['logfc']:.3f}\n"
                report += f"   Connected to: {genki_connections}\n\n"
        else:
            report += "No significant connection genes found.\n"
        
        report += """
---

## Matched Genes Analysis

"""
        
        # Add matched genes section
        for i, gene in enumerate(self.matched_genes, 1):
            genki_data = gene['genki_data']
            score = genki_data.get('dis', 'N/A')
            hit = genki_data.get('hit', 'N/A')
            
            report += f"""### {i}. {gene['genki']} (GenKI Rank {gene['genki_rank']})
**GenKI Score**: {score} | **Hit Rate**: {hit}% | **Found in**: {gene['found_in']}

**SHAP Connections:**
"""
            
            # Add SHAP connections for this gene
            gene_connections = self.shap_connections.get(gene['shap'], {})
            for direction, connections in gene_connections.items():
                if connections:
                    report += f"\n**{direction}** ({len(connections)} connections):\n"
                    for j, conn in enumerate(connections, 1):
                        # Check if this connection is significant
                        conn_gene_clean = conn['connected_gene']
                        if conn_gene_clean.endswith('_SIG'):
                            conn_gene_clean = conn_gene_clean.replace('_SIG', '')
                        is_significant = any(sc['connection_gene'].upper() == conn_gene_clean.upper() 
                                           for sc in self.significant_connection_genes)
                        significance_marker = " ⭐" if is_significant else ""
                        
                        report += f"{j}. {gene['shap']} ↔ {conn['connected_gene']} (SHAP: {conn['shap_value']:.4f}, {conn['connection_type']}, Cluster: {conn['cluster']}){significance_marker}\n"
                else:
                    report += f"\n**{direction}**: No connections found\n"
            
            report += "\n"
        
        # Add DEG overlap section
        report += """---

## Ground Truth DEG Validation

### Top Validated Molecules (by |logFC|)

"""
        
        # Sort by absolute log fold change
        top_degs = sorted(self.deg_overlap['matches'], key=lambda x: abs(x['logfc']), reverse=True)[:20]
        
        for i, deg in enumerate(top_degs, 1):
            direction = "UP" if deg['logfc'] > 0 else "DOWN"
            significance = " (SIGNIFICANT)" if deg['is_significant'] else ""
            report += f"{i}. **{deg['molecule']}** ({direction}) - logFC: {deg['logfc']:.3f}, p: {deg['pval']:.2e}, p-adj: {deg['pval_adj']:.2e}{significance}\n"
        
        # Add final gene list summary
        report += f"""

---

## Final Combined Gene List Summary

### Statistics
- **Total genes in final list**: {len(self.final_gene_list)}
- **GenKI genes**: {len([g for g in self.final_gene_list if g['source'] == 'GenKI'])}
- **SHAP connection genes**: {len([g for g in self.final_gene_list if g['source'] == 'SHAP_Connection'])}
- **Genes found in DEGs**: {len([g for g in self.final_gene_list if g['is_in_degs']])} ({len([g for g in self.final_gene_list if g['is_in_degs']])/len(self.final_gene_list)*100:.1f}%)
- **Significant genes (p < {self.pvalue_threshold})**: {len([g for g in self.final_gene_list if g['is_significant_deg']])} ({len([g for g in self.final_gene_list if g['is_significant_deg']])/len(self.final_gene_list)*100:.1f}%)

### Top 20 Genes by Significance (if available in DEGs)
"""
        
        # Show top genes from final list
        deg_genes_in_final = [g for g in self.final_gene_list if g['is_in_degs']]
        if deg_genes_in_final:
            # Sort by p-value for those with DEG info
            top_final_genes = sorted(deg_genes_in_final, key=lambda x: x['deg_info']['pval'])[:20]
            
            for i, gene in enumerate(top_final_genes, 1):
                deg_info = gene['deg_info']
                direction = "UP" if deg_info['logfc'] > 0 else "DOWN"
                source_info = f"GenKI (rank {gene['genki_rank']})" if gene['source'] == 'GenKI' else "SHAP Connection"
                significance = " ⭐" if deg_info['is_significant'] else ""
                
                report += f"{i}. **{gene['gene_name']}** ({direction}) - {source_info}\n"
                report += f"   p: {deg_info['pval']:.2e}, logFC: {deg_info['logfc']:.3f}{significance}\n"
                
                if gene['source'] == 'SHAP_Connection':
                    n_connections = gene['connection_info']['n_genki_connections']
                    report += f"   Connected to {n_connections} GenKI genes\n"
                report += "\n"
        
        # Add conclusions
        overall_performance = "Excellent" if self.deg_overlap['significant_coverage_percent'] > 35 else "Good" if self.deg_overlap['significant_coverage_percent'] > 25 else "Moderate"
        
        report += f"""

---

## Conclusions

### Overall Performance: {overall_performance}
- **{self.deg_overlap['significant_coverage_percent']:.1f}% significant validation rate** demonstrates strong biological relevance
- **{len(self.significant_connection_genes)} significant connection genes** identified through SHAP analysis
- **Enhanced analysis includes both Source and Target matching** plus significant connection identification
- **Final gene list combines direct predictions and network effects** for comprehensive coverage
- **GenKI predictions capture direct transcriptional effects** ({self.deg_overlap['genki_significant_percent']:.1f}% significant validation)
- **SHAP connections reveal network-level functional changes** with biological validation

### Biological Insights
- GenKI digital knockouts effectively predict direct gene expression changes
- SHAP feature importance identifies functionally relevant network connections  
- Combined approach captures both direct and indirect biological effects
- Significant connection genes represent validated network-level perturbation effects

---

*Report generated by Enhanced PerturbMap Analyzer with Significant Connection Gene Analysis*
*⭐ indicates genes with p-value < {self.pvalue_threshold}*
"""
        
        return report
    
    def save_results(self, output_dir: str = None) -> Dict[str, str]:
        """Save analysis results to files"""
        if output_dir is None:
            output_dir = f"enhanced_perturbmap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files_created = {}
        
        # Save markdown report
        report = self.generate_report()
        report_file = output_path / "analysis_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        files_created['report'] = str(report_file)
        
        # Save JSON data
        json_file = output_path / "analysis_data.json"
        with open(json_file, 'w') as f:
            json_data = self._convert_for_json(self.report_data)
            json.dump(json_data, f, indent=2, default=str)
        files_created['data'] = str(json_file)
        
        # NEW: Save final gene list as CSV
        final_genes_file = output_path / "final_gene_list.csv"
        if self.final_gene_list:
            # Flatten the final gene list data for CSV export
            flattened_final = []
            for gene in self.final_gene_list:
                flat_gene = {
                    'gene_name': gene['gene_name'],
                    'source': gene['source'],
                    'genki_rank': gene['genki_rank'],
                    'is_in_degs': gene['is_in_degs'],
                    'is_significant_deg': gene['is_significant_deg']
                }
                
                # Add DEG info if available
                if gene['deg_info']:
                    flat_gene.update({
                        'logfc': gene['deg_info']['logfc'],
                        'pval': gene['deg_info']['pval'],
                        'pval_adj': gene['deg_info']['pval_adj']
                    })
                else:
                    flat_gene.update({
                        'logfc': None,
                        'pval': None,
                        'pval_adj': None
                    })
                
                # Add GenKI info if available
                if gene['genki_data']:
                    flat_gene.update({
                        'genki_dis_score': gene['genki_data'].get('dis', None),
                        'genki_hit_rate': gene['genki_data'].get('hit', None)
                    })
                
                # Add connection info if available
                if gene.get('connection_info'):
                    flat_gene['n_genki_connections'] = gene['connection_info']['n_genki_connections']
                    related_genes = [rel['genki_gene'] for rel in gene['connection_info']['related_genki_genes']]
                    flat_gene['connected_to_genki_genes'] = '; '.join(related_genes)
                
                flattened_final.append(flat_gene)
            
            final_df = pd.DataFrame(flattened_final)
            final_df.to_csv(final_genes_file, index=False)
            files_created['final_genes'] = str(final_genes_file)
        
        # NEW: Save significant connection genes as CSV
        sig_connections_file = output_path / "significant_connection_genes.csv"
        if self.significant_connection_genes:
            flattened_connections = []
            for conn in self.significant_connection_genes:
                flat_conn = {
                    'connection_gene': conn['connection_gene'],
                    'original_deg_name': conn['original_deg_name'],
                    'logfc': conn['logfc'],
                    'pval': conn['pval'],
                    'pval_adj': conn['pval_adj'],
                    'score': conn['score'],
                    'n_genki_connections': conn['n_genki_connections']
                }
                
                # Add related GenKI genes info
                related_info = []
                for rel in conn['related_genki_genes']:
                    related_info.append(f"{rel['genki_gene']}({rel['direction']},{rel['shap_value']:.4f})")
                flat_conn['related_genki_genes'] = '; '.join(related_info)
                
                flattened_connections.append(flat_conn)
            
            connections_df = pd.DataFrame(flattened_connections)
            connections_df.to_csv(sig_connections_file, index=False)
            files_created['significant_connections'] = str(sig_connections_file)
        
        # Save unique molecules list
        molecules_file = output_path / "unique_molecules.txt"
        with open(molecules_file, 'w') as f:
            f.write(f"# Unique Molecules List\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total: {len(self.unique_molecules)} molecules\n\n")
            for i, molecule in enumerate(self.unique_molecules, 1):
                f.write(f"{i}. {molecule}\n")
        files_created['molecules'] = str(molecules_file)
        
        # Save DEG overlap results
        deg_file = output_path / "deg_overlap.csv" 
        if self.deg_overlap.get('matches'):
            deg_df = pd.DataFrame(self.deg_overlap['matches'])
            deg_df.to_csv(deg_file, index=False)
            files_created['deg_overlap'] = str(deg_file)
        
        # Save matched genes as CSV
        matched_genes_file = output_path / "matched_genes.csv"
        if self.matched_genes:
            # Flatten the matched genes data
            flattened_matches = []
            for match in self.matched_genes:
                flat_match = {
                    'genki_gene': match['genki'],
                    'shap_gene': match['shap'],
                    'genki_rank': match['genki_rank'],
                    'found_in': match['found_in'],
                    'dis_score': match['genki_data'].get('dis', 'N/A'),
                    'hit_rate': match['genki_data'].get('hit', 'N/A')
                }
                flattened_matches.append(flat_match)
            
            matched_df = pd.DataFrame(flattened_matches)
            matched_df.to_csv(matched_genes_file, index=False)
            files_created['matched_genes'] = str(matched_genes_file)
        
        # Save SHAP connections summary
        connections_file = output_path / "shap_connections_summary.csv"
        if self.shap_connections:
            connection_rows = []
            for gene, directions in self.shap_connections.items():
                for direction, connections in directions.items():
                    for conn in connections:
                        # Check if this connection is significant
                        conn_gene_clean = conn['connected_gene']
                        if conn_gene_clean.endswith('_SIG'):
                            conn_gene_clean = conn_gene_clean.replace('_SIG', '')
                        is_significant = any(sc['connection_gene'].upper() == conn_gene_clean.upper() 
                                           for sc in self.significant_connection_genes)
                        
                        connection_rows.append({
                            'gene': gene,
                            'direction': direction,
                            'connected_gene': conn['connected_gene'],
                            'shap_value': conn['shap_value'],
                            'cluster': conn['cluster'],
                            'connection_type': conn['connection_type'],
                            'is_significant_deg': is_significant
                        })
            
            if connection_rows:
                connections_df = pd.DataFrame(connection_rows)
                connections_df.to_csv(connections_file, index=False)
                files_created['connections'] = str(connections_file)
        
        logger.info(f"Results saved to {output_path}/")
        for file_type, file_path in files_created.items():
            logger.info(f"  {file_type}: {file_path}")
        
        return files_created
    
    def _convert_for_json(self, obj):
        """Convert numpy types and other non-serializable objects for JSON"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def run_full_analysis(self, output_dir: str = None) -> bool:
        """Run complete analysis pipeline"""
        logger.info("Starting enhanced PerturbMap analysis pipeline...")
        
        try:
            # Load data
            if not self.load_data():
                return False
            
            # Run analysis steps
            self.find_matched_genes()
            self.extract_shap_connections()
            self.identify_significant_connection_genes()  # NEW
            self.create_final_gene_list()  # NEW
            self.create_unique_molecules_list()
            self.analyze_deg_overlap()
            
            # Save results
            files_created = self.save_results(output_dir)
            
            # Print summary to console
            self._print_summary()
            
            logger.info("Enhanced analysis pipeline finished successfully!")
            logger.info(f"Results saved to: {list(files_created.values())[0].split('/')[0]}/")
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced analysis pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_summary(self):
        """Print analysis summary to console"""
        print("\n" + "="*80)
        print("ENHANCED PERTURBMAP ANALYSIS SUMMARY")
        print("="*80)
        print(f"Data Loaded:")
        print(f"   SHAP entries: {len(self.shap_data):,}")
        print(f"   GenKI genes: {len(self.genki_data):,}")
        print(f"   Ground truth DEGs: {len(self.deg_data):,}")
        print(f"   Significant DEGs (p < {self.pvalue_threshold}): {len(self.deg_data[self.deg_data['pvals'] < self.pvalue_threshold]):,}")
        
        print(f"\nGene Matching (Source & Target):")
        print(f"   Matched genes: {len(self.matched_genes)} ({len(self.matched_genes)/len(self.genki_data)*100:.1f}% of GenKI)")
        
        # Breakdown by location
        source_matches = [m for m in self.matched_genes if 'source' in m['found_in']]
        target_matches = [m for m in self.matched_genes if 'target' in m['found_in']]
        both_matches = [m for m in self.matched_genes if 'source' in m['found_in'] and 'target' in m['found_in']]
        
        print(f"   - Found in sources: {len(source_matches)}")
        print(f"   - Found in targets: {len(target_matches)}")
        print(f"   - Found in both: {len(both_matches)}")
        
        print(f"\nNEW: Significant Connection Genes:")
        print(f"   Connection genes with p < {self.pvalue_threshold}: {len(self.significant_connection_genes)}")
        if self.significant_connection_genes:
            top_3_connections = sorted(self.significant_connection_genes, key=lambda x: x['pval'])[:3]
            for i, conn in enumerate(top_3_connections, 1):
                print(f"   {i}. {conn['connection_gene']} (p: {conn['pval']:.2e}, logFC: {conn['logfc']:.3f})")
        
        print(f"\nNEW: Final Combined Gene List:")
        print(f"   Total genes: {len(self.final_gene_list)}")
        print(f"   - GenKI genes: {len([g for g in self.final_gene_list if g['source'] == 'GenKI'])}")
        print(f"   - SHAP connection genes: {len([g for g in self.final_gene_list if g['source'] == 'SHAP_Connection'])}")
        print(f"   - Found in DEGs: {len([g for g in self.final_gene_list if g['is_in_degs']])} ({len([g for g in self.final_gene_list if g['is_in_degs']])/len(self.final_gene_list)*100:.1f}%)")
        print(f"   - Significant DEGs: {len([g for g in self.final_gene_list if g['is_significant_deg']])} ({len([g for g in self.final_gene_list if g['is_significant_deg']])/len(self.final_gene_list)*100:.1f}%)")
        
        print(f"\nGround Truth Validation:")
        print(f"   Total molecules analyzed: {self.deg_overlap['total_molecules']}")
        print(f"   Found in DEGs: {self.deg_overlap['total_matches']} ({self.deg_overlap['coverage_percent']:.1f}%)")
        print(f"   Significant DEGs: {self.deg_overlap['significant_matches']} ({self.deg_overlap['significant_coverage_percent']:.1f}%)")
        print(f"   GenKI validation: {self.deg_overlap['genki_matches']}/{self.deg_overlap['genki_total']} ({self.deg_overlap['genki_percent']:.1f}%)")
        print(f"   GenKI significant: {self.deg_overlap['genki_significant_matches']}/{self.deg_overlap['genki_total']} ({self.deg_overlap['genki_significant_percent']:.1f}%)")
        print(f"   SHAP validation: {self.deg_overlap['shap_matches']} targets ({self.deg_overlap['shap_significant_matches']} significant)")
        
        # Show top validated genes
        if self.deg_overlap.get('matches'):
            top_genes = sorted([m for m in self.deg_overlap['matches'] if m['is_significant']], 
                             key=lambda x: abs(x['logfc']), reverse=True)[:5]
            if top_genes:
                print(f"\nTop Significant Validated Genes:")
                for gene in top_genes:
                    direction = "UP" if gene['logfc'] > 0 else "DOWN"
                    print(f"   {gene['molecule']} ({direction}) - logFC: {gene['logfc']:.3f}, p: {gene['pval']:.2e}")
        
        # Show matched genes details
        print(f"\nMatched Genes Details:")
        for i, gene in enumerate(self.matched_genes, 1):
            print(f"{i:2d}. {gene['genki']} <-> {gene['shap']} (found in: {gene['found_in']}, rank: {gene['genki_rank']})")
        
        print("="*80)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Enhanced PerturbMap Analysis Pipeline - Compare GenKI, SHAP, and ground truth DEGs with significant connection analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default parameters
  python enhanced_perturbmap_analyzer.py --shap_file shap.csv --genki_file genki.csv --deg_file degs.csv
  
  # Analysis with custom parameters
  python enhanced_perturbmap_analyzer.py --shap_file shap.csv --genki_file genki.csv --deg_file degs.csv --top_n 20 --pvalue_threshold 0.01 --output_dir my_results
        """
    )
    
    # Required arguments
    parser.add_argument('--shap_file', type=str, required=True, help='Path to SHAP feature importance CSV file')
    parser.add_argument('--genki_file', type=str, required=True, help='Path to GenKI digital knockout results CSV file')
    parser.add_argument('--deg_file', type=str, required=True, help='Path to ground truth DEGs CSV file')
    
    # Optional arguments
    parser.add_argument('--top_n', type=int, default=5, help='Number of top SHAP connections to extract for each gene (default: 5)')
    parser.add_argument('--pvalue_threshold', type=float, default=0.05, help='P-value threshold for significant DEGs (default: 0.05)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name (default: enhanced_perturbmap_analysis_TIMESTAMP)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input files
    for file_path in [args.shap_file, args.genki_file, args.deg_file]:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return 1
    
    # Validate p-value threshold
    if not 0 < args.pvalue_threshold < 1:
        logger.error(f"P-value threshold must be between 0 and 1, got: {args.pvalue_threshold}")
        return 1
    
    # Run analysis
    analyzer = EnhancedPerturbMapAnalyzer(
        shap_file=args.shap_file,
        genki_file=args.genki_file, 
        deg_file=args.deg_file,
        top_n=args.top_n,
        pvalue_threshold=args.pvalue_threshold
    )
    
    success = analyzer.run_full_analysis(output_dir=args.output_dir)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())