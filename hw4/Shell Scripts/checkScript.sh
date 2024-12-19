echo "Checking all outputs"

diff /scratch/ualclsd0167/output.5000.5000.1.txt /scratch/ualclsd0167/output.5000.5000.2.txt
diff /scratch/ualclsd0167/output.5000.5000.1.txt /scratch/ualclsd0167/output.5000.5000.4.txt
diff /scratch/ualclsd0167/output.5000.5000.1.txt /scratch/ualclsd0167/output.5000.5000.8.txt
diff /scratch/ualclsd0167/output.5000.5000.1.txt /scratch/ualclsd0167/output.5000.5000.10.txt
diff /scratch/ualclsd0167/output.5000.5000.1.txt /scratch/ualclsd0167/output.5000.5000.16.txt
diff /scratch/ualclsd0167/output.5000.5000.1.txt /scratch/ualclsd0167/output.5000.5000.20.txt

diff /scratch/ualclsd0167/output.5000.5000.1.txt /scratch/ualclsd0167/noblockres/output.5000.5000.1.txt
diff /scratch/ualclsd0167/noblockres/output.5000.5000.1.txt /scratch/ualclsd0167/noblockres/output.5000.5000.2.txt
diff /scratch/ualclsd0167/noblockres/output.5000.5000.1.txt /scratch/ualclsd0167/noblockres/output.5000.5000.4.txt
diff /scratch/ualclsd0167/noblockres/output.5000.5000.1.txt /scratch/ualclsd0167/noblockres/output.5000.5000.8.txt
diff /scratch/ualclsd0167/noblockres/output.5000.5000.1.txt /scratch/ualclsd0167/noblockres/output.5000.5000.10.txt
diff /scratch/ualclsd0167/noblockres/output.5000.5000.1.txt /scratch/ualclsd0167/noblockres/output.5000.5000.16.txt
diff /scratch/ualclsd0167/noblockres/output.5000.5000.1.txt /scratch/ualclsd0167/noblockres/output.5000.5000.20.txt

echo "All outputs are correct if nothing has shown up yet!"
