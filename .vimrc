set nocompatible
filetype on
filetype plugin on
filetype indent on
syntax on
set number
set shiftwidth=2
set expandtab
set nowrap
set incsearch
set ignorecase
set smartcase
set showcmd
set showmode
set showmatch
set hlsearch
set history=1000
set wildmenu
set wildmode=list:longest
set wildignore=*.docx,*.jpg,*.png,*.gif,*.pdf,*.pyc,*.exe,*.flv,*.img,*.xlsx

set autoindent
set smartindent
set tabstop=2

filetype plugin indent on

colorscheme space

set vb
set shellcmdflag=-ic

autocmd BufRead,BufNewFile *.note set syntax=python
autocmd BufRead,BufNewFile *.py set syntax=python
