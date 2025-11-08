# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]; then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
if [ -d ~/.bashrc.d ]; then
    for rc in ~/.bashrc.d/*; do
        if [ -f "$rc" ]; then
            . "$rc"
        fi
    done
fi
unset rc

alias cll='clear;ls -l'
alias clla='clear;ls -la'
alias clc='clear'
alias src='source ~/.bashrc'
alias py='python'

alias lla='ls -la'

alias ebash='vim ~/.bashrc'
alias evim='vim ~/.vimrc'

# Working shortcuts
alias dev='cd /home/tj/dev/'
alias pract='cd /home/tj/dev/gnc/practice'

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/tj/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/tj/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/tj/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/tj/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

