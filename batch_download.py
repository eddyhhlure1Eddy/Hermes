"""
Batch Download A-Share Historical Data

Author: eddy
"""

import os
import sys
import time
import json
import pandas as pd
from datetime import datetime
from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams
from tqdm import tqdm

class FullHistoryCollector:
    """Full A-Share Historical Data Collector"""

    def __init__(self, output_dir="full_stock_data"):
        self.output_dir = output_dir
        self.training_dir = os.path.join(output_dir, "training_data")
        self.metadata_file = os.path.join(output_dir, "metadata.json")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)

        self.api = None
        self.connected = False
        self.metadata = self.load_metadata()

    def load_metadata(self):
        """Load metadata"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'downloaded_stocks': [],
            'failed_stocks': [],
            'last_update': None,
            'total_records': 0
        }

    def save_metadata(self):
        """Save metadata"""
        self.metadata['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def connect(self):
        """Connect to TDX server"""
        servers = [
            ("124.71.187.122", 7709),
            ("180.153.18.170", 7709),
            ("180.153.18.171", 7709),
        ]

        self.api = TdxHq_API()

        for ip, port in servers:
            try:
                if self.api.connect(ip, port):
                    self.connected = True
                    return True
            except:
                continue

        return False

    def generate_all_stock_codes(self):
        """Generate all possible A-share stock codes"""
        stocks = []

        sh_prefixes = ['600', '601', '603', '605', '688']
        for prefix in sh_prefixes:
            for i in range(1000):
                code = f"{prefix}{i:03d}"
                stocks.append({
                    'code': code,
                    'market': 1,
                    'name': f'SH{code}',
                    'exchange': 'SH'
                })

        sz_prefixes = ['000', '001', '002', '003', '300', '301']
        for prefix in sz_prefixes:
            for i in range(1000):
                code = f"{prefix}{i:03d}"
                stocks.append({
                    'code': code,
                    'market': 0,
                    'name': f'SZ{code}',
                    'exchange': 'SZ'
                })

        return stocks

    def get_stock_bars(self, code, market, max_count=10000):
        """Get historical K-line data from 1999"""
        if not self.connected:
            return None

        try:
            category = TDXParams.KLINE_TYPE_DAILY

            all_data = []
            pos = 0
            batch_size = 800

            while pos < max_count:
                data = self.api.get_security_bars(category, market, code, pos, batch_size)

                if data is None or len(data) == 0:
                    break

                all_data.extend(data)
                pos += batch_size

                if len(data) < batch_size:
                    break

                time.sleep(0.02)

            if not all_data:
                return None

            df = pd.DataFrame(all_data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')

            df = df.rename(columns={
                'datetime': 'date',
                'vol': 'volume'
            })

            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

            return df

        except:
            return None

def print_status(collector):
    """Print current download status"""
    print(f"\nCurrent Status:")
    print(f"  Downloaded: {len(collector.metadata['downloaded_stocks'])} stocks")
    print(f"  Failed: {len(collector.metadata['failed_stocks'])} stocks")
    print(f"  Total records: {collector.metadata['total_records']:,}")
    if collector.metadata['last_update']:
        print(f"  Last update: {collector.metadata['last_update']}")

def batch_download():
    """Batch download with progress tracking"""

    print("="*70)
    print("A-Share Historical Data Batch Download System")
    print("="*70)

    collector = FullHistoryCollector(output_dir='full_stock_data')

    print_status(collector)
    
    print(f"\nBatch Download Strategy:")
    print(f"  Total potential stocks: ~11,000")
    print(f"  Batch size: 500 stocks per batch")
    print(f"  Estimated time per batch: ~20 minutes")
    print(f"  Total estimated time: ~7-8 hours")
    
    batches = [
        {"name": "Batch 1: SH 600000-600499", "start": 0, "count": 500},
        {"name": "Batch 2: SH 600500-600999", "start": 500, "count": 500},
        {"name": "Batch 3: SH 601000-601499", "start": 1000, "count": 500},
        {"name": "Batch 4: SH 601500-603999", "start": 1500, "count": 500},
        {"name": "Batch 5: SH 604000-605999", "start": 2000, "count": 500},
        {"name": "Batch 6: SH 606000-688999", "start": 2500, "count": 500},
        {"name": "Batch 7: SZ 000000-000499", "start": 5000, "count": 500},
        {"name": "Batch 8: SZ 000500-000999", "start": 5500, "count": 500},
        {"name": "Batch 9: SZ 001000-001499", "start": 6000, "count": 500},
        {"name": "Batch 10: SZ 001500-002499", "start": 6500, "count": 500},
        {"name": "Batch 11: SZ 002500-002999", "start": 7000, "count": 500},
        {"name": "Batch 12: SZ 003000-003999", "start": 7500, "count": 500},
        {"name": "Batch 13: SZ 300000-300499", "start": 8000, "count": 500},
        {"name": "Batch 14: SZ 300500-300999", "start": 8500, "count": 500},
        {"name": "Batch 15: SZ 301000-301999", "start": 9000, "count": 500},
    ]
    
    print(f"\nTotal batches: {len(batches)}")
    print(f"\nPress Enter to start batch download, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled by user")
        return
    
    overall_start = datetime.now()
    
    for i, batch in enumerate(batches, 1):
        print(f"\n{'='*70}")
        print(f"Starting {batch['name']} ({i}/{len(batches)})")
        print(f"{'='*70}")
        
        batch_start = datetime.now()
        
        stock_list = collector.generate_all_stock_codes()
        batch_stocks = stock_list[batch['start']:batch['start'] + batch['count']]
        
        if not collector.connect():
            print(f"Failed to connect to TDX server, retrying in 30 seconds...")
            time.sleep(30)
            if not collector.connect():
                print(f"Failed to connect, skipping batch {i}")
                continue
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        from tqdm import tqdm
        
        for stock in tqdm(batch_stocks, desc=f"Batch {i}"):
            code = stock['code']
            market = stock['market']
            exchange = stock['exchange']
            stock_id = f"{exchange}{code}"
            
            if stock_id in collector.metadata['downloaded_stocks']:
                skipped_count += 1
                continue
            
            try:
                df = collector.get_stock_bars(code, market)
                
                if df is not None and len(df) >= 100:
                    filename = f"{exchange}{code}.csv"
                    filepath = os.path.join(collector.output_dir, filename)
                    df.to_csv(filepath, index=False, encoding='utf-8-sig')
                    
                    training_path = os.path.join(collector.training_dir, filename)
                    df.to_csv(training_path, index=False)
                    
                    collector.metadata['downloaded_stocks'].append(stock_id)
                    collector.metadata['total_records'] += len(df)
                    
                    success_count += 1
                else:
                    collector.metadata['failed_stocks'].append(stock_id)
                    failed_count += 1
                
                if success_count % 50 == 0:
                    collector.save_metadata()
                
                time.sleep(0.05)
                
            except Exception as e:
                collector.metadata['failed_stocks'].append(stock_id)
                failed_count += 1
        
        if collector.api:
            collector.api.disconnect()
        
        collector.save_metadata()
        
        batch_end = datetime.now()
        batch_duration = (batch_end - batch_start).total_seconds()
        
        print(f"\n{batch['name']} completed:")
        print(f"  Success: {success_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Duration: {batch_duration:.0f} seconds ({batch_duration/60:.1f} minutes)")
        
        if i < len(batches):
            print(f"\nWaiting 10 seconds before next batch...")
            time.sleep(10)
    
    overall_end = datetime.now()
    overall_duration = (overall_end - overall_start).total_seconds()
    
    print(f"\n{'='*70}")
    print(f"All batches completed!")
    print(f"{'='*70}")
    print(f"Total downloaded: {len(collector.metadata['downloaded_stocks'])} stocks")
    print(f"Total records: {collector.metadata['total_records']:,}")
    print(f"Total duration: {overall_duration:.0f} seconds ({overall_duration/3600:.2f} hours)")
    print(f"Output directory: {collector.output_dir}")
    print(f"Training directory: {collector.training_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    try:
        batch_download()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        print("Progress has been saved, you can resume later")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

