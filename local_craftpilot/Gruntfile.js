module.exports = function(grunt) {
    grunt.initConfig({
        babel: {
            options: {
                sourceMap: true,
                presets: [['@babel/preset-env', {
                    targets: {
                        browsers: ['last 2 versions']
                    },
                    modules: 'amd'
                }]]
            },
            dist: {
                files: [{
                    expand: true,
                    cwd: 'amd/src',
                    src: ['**/*.js'],
                    dest: 'amd/build',
                    ext: '.min.js'
                }]
            }
        },
        watch: {
            amd: {
                files: ['amd/src/**/*.js'],
                tasks: ['babel'],
                options: {
                    spawn: false
                }
            }
        }
    });

    grunt.loadNpmTasks('grunt-babel');
    grunt.loadNpmTasks('grunt-contrib-watch');

    grunt.registerTask('default', ['babel']);
    grunt.registerTask('dev', ['watch']);
};