DFILES = duan2009/duan.SC.10000.raw_sub.matrix \
         duan2009/duan.SC.10000.raw_sub.bed \


all: $(DFILES)

duan2009/%:
	mkdir -p duan2009
	wget http://cbio.ensmp.fr/~nvaroquaux/data/$@ -O $@

clean:
	rm -f *.pdb *.pdb.txt *.pdb.temp.txt *.pdb.temp.pdb *.sh
	rm -f *.log
	rm -f iterate.dat
	rm -rf duan2009
	rm -f *.structure
	rm -f counts.txt wish_distances.txt  ipopt.out
