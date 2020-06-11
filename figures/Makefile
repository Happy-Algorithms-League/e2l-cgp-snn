all: fig1 fig2 fig3 fig4 fig5
fig1: Fig1_intro.pdf
fig2: Fig2_cgp_sketch.pdf
fig3: Fig3_reward_driven.pdf
fig4: Fig4_error_driven.pdf
fig5: Fig5_corr_learning.pdf

Fig3_reward_driven.eps: Fig3_reward_driven.py ../data/reward-driven/9db4731418d083fcd3d0d9528f805aaa/*
	python Fig3_reward_driven.py ../data/reward-driven/9db4731418d083fcd3d0d9528f805aaa/

Fig4_error_driven.eps: Fig4_error_driven.py ../data/error-driven/91f21d6d01014991285e5264a59aa967/*
	python Fig4_error_driven.py ../data/error-driven/91f21d6d01014991285e5264a59aa967/

Fig5_corr_learning.eps: Fig5_corr_learning.py ../data/correlation_based/6a8d0387281207f8c8ed345d40a417fb
	python Fig5_corr_learning.py ../data/correlation_based/6a8d0387281207f8c8ed345d40a417fb

%.pdf : %.eps
	epspdf $<
